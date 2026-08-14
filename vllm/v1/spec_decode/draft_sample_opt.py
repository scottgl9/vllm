# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Optimized draft token sampling for MTP speculative decoding.

The lm_head matmul (hidden_size x vocab_size) dominates MTP per-step cost
(~85%, ~5.6ms for 248K vocab at bf16 on GB10). These optimizations target
reducing that cost:

1. compiled_greedy_sample: torch.compile wrapper (3.3x speedup, 100% accuracy)
2. FP8LMHeadSampler: FP8 weight quantization (9.5x speedup, ~85% accuracy)
3. quantize_mtp_moe_fp8: post-quantize MTP MoE expert weights to FP8 at load
   time (before CUDA graph capture). Halves active-expert memory per draft step.
   Enable with VLLM_MTP_MOE_FP8=1.

Usage: set VLLM_DRAFT_SAMPLE_OPT=compiled|fp8 environment variable.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Option 1: torch.compile'd greedy sample
# ---------------------------------------------------------------------------

def _make_compiled_sample_fn():
    """Create a torch.compile'd function for lm_head matmul + argmax."""

    @torch.compile(dynamic=False)
    def _compiled_fn(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        org_vocab_size: int,
    ) -> torch.Tensor:
        # matmul: [batch, hidden] @ [vocab, hidden]^T -> [batch, vocab]
        logits = F.linear(hidden_states, weight)
        # trim padding and argmax (Inductor may fuse these)
        logits = logits[..., :org_vocab_size]
        return logits.argmax(dim=-1)

    return _compiled_fn


_compiled_sample_fn = None


def compiled_greedy_sample(
    lm_head: torch.nn.Module,
    logits_processor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Greedy sample using torch.compile'd matmul+argmax.

    Benefits: Inductor fuses the vocab trim + argmax reduction,
    avoiding a separate kernel launch for argmax. ~3.3x speedup on GB10.
    """
    global _compiled_sample_fn
    if _compiled_sample_fn is None:
        _compiled_sample_fn = _make_compiled_sample_fn()

    return _compiled_sample_fn(
        hidden_states,
        lm_head.weight,
        logits_processor.org_vocab_size,
    )


# ---------------------------------------------------------------------------
# Option 2: FP8 lm_head quantization
# ---------------------------------------------------------------------------

class FP8LMHeadSampler:
    """Wraps a bf16 lm_head with FP8 weights for faster draft sampling.

    Quantizes the lm_head weight matrix from bf16 (~1 GB for 248K vocab)
    to float8_e4m3fn (~500 MB), halving memory bandwidth for the matmul.
    ~9.5x speedup on GB10. Token accuracy ~85% with per-tensor quantization
    (sufficient for draft tokens; verification catches mismatches).
    """

    def __init__(self, lm_head: torch.nn.Module, org_vocab_size: int):
        weight = lm_head.weight.data  # [vocab_size_padded, hidden_size]

        fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0

        # Per-tensor quantization for weight
        self.weight_scale_val = weight.abs().max().item() / fp8_max
        self.weight_fp8 = (weight / self.weight_scale_val).to(
            torch.float8_e4m3fn
        )
        self.scale_b = torch.tensor(
            self.weight_scale_val,
            dtype=torch.float32,
            device=weight.device,
        )
        self.org_vocab_size = org_vocab_size
        self.fp8_max = fp8_max

        # Memory savings log
        orig_mb = weight.numel() * weight.element_size() / 1024 / 1024
        fp8_mb = self.weight_fp8.numel() / 1024 / 1024  # 1 byte per element
        logger.info(
            "FP8LMHeadSampler: lm_head %.0f MB (bf16) -> %.0f MB (fp8), "
            "-%.0f%% memory, vocab=%d, hidden=%d",
            orig_mb,
            fp8_mb,
            (1 - fp8_mb / orig_mb) * 100,
            weight.shape[0],
            weight.shape[1],
        )

    def sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute draft token IDs using FP8 lm_head matmul + argmax."""
        x = hidden_states.view(-1, hidden_states.shape[-1])

        # Dynamic per-tensor quantization of input activations
        scale_a_val = x.abs().max().item() / self.fp8_max
        scale_a = torch.tensor(
            scale_a_val, dtype=torch.float32, device=x.device
        )
        x_fp8 = (x / scale_a_val).to(torch.float8_e4m3fn)

        # FP8 matmul: [batch, hidden] @ [hidden, vocab]
        logits = torch._scaled_mm(
            x_fp8,
            self.weight_fp8.t(),
            out_dtype=hidden_states.dtype,
            scale_a=scale_a,
            scale_b=self.scale_b,
        )

        # Trim vocab padding and argmax
        logits = logits[..., : self.org_vocab_size]
        return logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Option 3: FP8 post-quantization of MTP linear layers (fc, attn, mlp)
# ---------------------------------------------------------------------------


class Fp8PostQuantLinearMethod:
    """Minimal quant_method adapter for post-quantized FP8 linear layers.

    Installed by ``post_quantize_linear_to_fp8`` in place of the layer's
    original quant_method. The layer keeps its original class (e.g.
    ColumnParallelLinear) and forward() unchanged -- only .weight,
    .weight_scale, and .quant_method are swapped -- so return_bias,
    gather_output, and skip_bias_add semantics all continue to work exactly
    as before. Mirrors Fp4PostQuantLinearMethod in nvfp4_post_quant.py.

    Quantizes weight to float8_e4m3fn with a single per-tensor scale;
    activations are dynamically quantized per-call via torch._scaled_mm().
    """

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        scale_a_val = x.abs().max().clamp(min=1e-12).item() / fp8_max
        scale_a = torch.tensor(scale_a_val, dtype=torch.float32, device=x.device)
        x_fp8 = (x / scale_a_val).to(torch.float8_e4m3fn)

        out = torch._scaled_mm(
            x_fp8,
            layer.weight.t(),
            out_dtype=x.dtype if x.dtype != torch.float8_e4m3fn else torch.bfloat16,
            scale_a=scale_a,
            scale_b=layer.weight_scale,
        )
        if bias is not None:
            out = out + bias
        return out.view(*orig_shape[:-1], out.shape[-1])

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass  # weights are already prepared


def post_quantize_linear_to_fp8(layer: torch.nn.Module, name: str) -> bool:
    """Post-quantize a BF16/FP16 linear layer's weight to FP8 in-place.

    Returns True if quantization was applied, False if skipped (weight is
    already quantized, or not BF16/FP16).
    """
    weight = layer.weight.data
    if weight.dtype not in (torch.bfloat16, torch.float16):
        logger.debug(
            "Skipping FP8 post-quant for %s: dtype is %s (not BF16/FP16)",
            name,
            weight.dtype,
        )
        return False

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    scale_val = weight.abs().max().clamp(min=1e-12).item() / fp8_max
    weight_scale = torch.tensor(scale_val, dtype=torch.float32, device=weight.device)
    weight_fp8 = (weight / scale_val).clamp(-fp8_max, fp8_max).to(fp8_dtype)

    del layer.weight
    layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    layer.quant_method = Fp8PostQuantLinearMethod()

    orig_mb = weight.numel() * weight.element_size() / 1024 / 1024
    fp8_mb = weight_fp8.numel() / 1024 / 1024
    logger.info(
        "FP8 post-quantized %s: %.1f MB (bf16) -> %.1f MB (fp8)",
        name, orig_mb, fp8_mb,
    )
    return True


def apply_fp8_post_quant(
    model: torch.nn.Module,
    layer_patterns: list[str],
    name_filter: Callable[[str], bool] | None = None,
) -> int:
    """Post-quantize matching linear layers to FP8 in-place.

    Walks model.named_modules() and converts any BF16/FP16 linear layer
    whose name ends with a suffix in ``layer_patterns`` to FP8. Matches on
    hasattr(module, "weight") rather than isinstance(module, torch.nn.Linear)
    because vLLM's parallel linear layers (ColumnParallelLinear,
    RowParallelLinear, QKVParallelLinear, ...) subclass nn.Module via
    LinearBase/PluggableLayer, not nn.Linear -- an isinstance(nn.Linear)
    check silently matches zero layers on any real vLLM model.

    Args:
        model: The model to quantize.
        layer_patterns: List of name suffixes to match (e.g., ["fc", "lm_head"]).
        name_filter: Optional predicate on the module's dotted name; only
            matching layers are quantized. Use this to scope quantization to
            a submodule (e.g. ``lambda n: n.startswith("mtp.")``) since
            layer_patterns like "q_proj"/"gate_proj" would otherwise also
            match the main model's own (already-quantized) layers.

    Returns:
        Number of layers converted.
    """
    count = 0

    for name, module in list(model.named_modules()):
        if not hasattr(module, "weight"):
            continue
        if not any(name.endswith(pat) for pat in layer_patterns):
            continue
        if name_filter is not None and not name_filter(name):
            continue
        if post_quantize_linear_to_fp8(module, name):
            count += 1

    return count


def maybe_quantize_mtp_fp8(draft_model: torch.nn.Module) -> None:
    """Post-quantize MTP linear layers and MoE experts to FP8.

    Called when VLLM_MTP_FP8=1. Converts, scoped to the mtp.* submodule only:
    - fc, self_attn q/k/v/o proj, and mlp gate/up/down proj to FP8 (via
      apply_fp8_post_quant)
    - MoE expert weights to FP8 (via quantize_mtp_moe_fp8, skipped on SM120+)

    Must be called before CUDA graph capture.
    """
    from vllm.platforms import current_platform

    n_fc = apply_fp8_post_quant(
        draft_model,
        [
            "fc",
            "lm_head",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        name_filter=lambda n: n.startswith("mtp."),
    )
    logger.info(
        "maybe_quantize_mtp_fp8: converted %d MTP linear layer(s) to FP8",
        n_fc,
    )

    # MoE FP8 is skipped on SM120+ (Triton limitation)
    if current_platform.has_device_capability(120):
        logger.info(
            "maybe_quantize_mtp_fp8: skipping MoE FP8 on SM120+ "
            "(use VLLM_MTP_MOE_FP8=1 separately if needed)"
        )
        return

    n_moe = quantize_mtp_moe_fp8(draft_model)
    logger.info(
        "maybe_quantize_mtp_fp8: converted %d MoE layer(s) to FP8",
        n_moe,
    )


# ---------------------------------------------------------------------------
# Option 4: FP8 post-quantization of MTP MoE expert weights
# ---------------------------------------------------------------------------

def quantize_mtp_moe_fp8(
    mtp_model: torch.nn.Module,
    name_filter: Callable[[str], bool] | None = None,
) -> int:
    """Post-quantize MoE expert weights from bf16 to FP8 W8A8.

    Must be called BEFORE CUDA graph capture so the graphs capture the FP8
    kernel path. Replaces quant_method.moe_quant_config and quant_method.kernel
    on each UnquantizedFusedMoEMethod layer whose weights are still bf16.

    Per-expert per-tensor quantization: one scale per expert per weight matrix.
    Halves active-expert memory bandwidth per draft step:
      35B: 50 MB -> 25 MB/step, 122B: 151 MB -> 75 MB/step.

    Args:
        mtp_model: Model (or submodule) to walk for MoE layers.
        name_filter: Optional predicate on the module's dotted name (as
            yielded by ``named_modules()``); only matching layers are
            quantized. ``None`` (default) quantizes every eligible layer,
            preserving the original MTP-only behavior.

    Returns:
        The number of MoE layers quantized.
    """
    from vllm.model_executor.layers.fused_moe.config import (
        fp8_w8a8_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        make_unquantized_moe_kernel,
    )
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )
    from vllm.model_executor.utils import replace_parameter

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max  # 448.0
    count = 0

    for name, layer in mtp_model.named_modules():
        if name_filter is not None and not name_filter(name):
            continue
        qm = getattr(layer, "quant_method", None)
        if not isinstance(qm, UnquantizedFusedMoEMethod):
            continue
        if not (hasattr(layer, "w13_weight") and hasattr(layer, "w2_weight")):
            continue
        if layer.w13_weight.dtype != torch.bfloat16:
            continue

        num_experts = layer.w13_weight.shape[0]
        w13 = layer.w13_weight.data.float()
        w2 = layer.w2_weight.data.float()

        # Per-expert per-tensor scales
        w13_scale = (
            w13.abs().view(num_experts, -1).max(dim=1).values / fp8_max
        ).clamp(min=1e-12)
        w2_scale = (
            w2.abs().view(num_experts, -1).max(dim=1).values / fp8_max
        ).clamp(min=1e-12)

        w13_fp8 = (w13 / w13_scale.view(-1, 1, 1)).clamp(-fp8_max, fp8_max).to(
            fp8_dtype
        )
        w2_fp8 = (w2 / w2_scale.view(-1, 1, 1)).clamp(-fp8_max, fp8_max).to(
            fp8_dtype
        )

        replace_parameter(
            layer, "w13_weight", torch.nn.Parameter(w13_fp8, requires_grad=False)
        )
        replace_parameter(
            layer, "w2_weight", torch.nn.Parameter(w2_fp8, requires_grad=False)
        )

        fp8_qconfig = fp8_w8a8_moe_quant_config(
            w1_scale=w13_scale, w2_scale=w2_scale
        )
        qm.moe_quant_config = fp8_qconfig
        qm.kernel = make_unquantized_moe_kernel(
            backend=qm.unquantized_backend,
            quant_config=fp8_qconfig,
            moe_config=qm.moe,
        )
        count += 1

        orig_mb = (w13.numel() + w2.numel()) * 4 / 1024 / 1024  # float32 ref
        fp8_mb = (w13_fp8.numel() + w2_fp8.numel()) / 1024 / 1024
        logger.info(
            "quantize_mtp_moe_fp8: %s -> FP8, %.0f MB (bf16) -> %.0f MB (fp8), "
            "num_experts=%d",
            name,
            orig_mb / 2,  # bf16 is 2 bytes
            fp8_mb,
            num_experts,
        )

    return count
