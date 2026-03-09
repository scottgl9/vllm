# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic NVFP4 post-quantization utility for GDN linear layers.

Converts BF16/FP16 linear layers to NVFP4 at load time on SM120+ GPUs,
reducing memory bandwidth ~2x for large attention projections (e.g., in_proj_qkvz).
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.nn import Parameter

from vllm.platforms import current_platform

logger = logging.getLogger(__name__)

_FP4_MAX = 6.0


class Fp4PostQuantLinearMethod:
    """Minimal quant_method adapter for post-quantized NVFP4 linear layers.

    Installed by ``post_quantize_linear_to_nvfp4`` after packing the BF16
    weight into NVFP4 format. Dispatches ``apply`` to ``apply_nvfp4_linear``.
    """

    def __init__(self, backend):
        self.backend = backend

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
            apply_nvfp4_linear,
        )

        return apply_nvfp4_linear(
            backend=self.backend,
            layer=layer,
            x=x,
            bias=bias,
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        pass  # weights are already prepared


def post_quantize_linear_to_nvfp4(layer: nn.Module, name: str, backend) -> bool:
    """Post-quantize a BF16/FP16 linear layer to NVFP4 in-place.

    Converts the dense BF16 weight into packed FP4 uint8 format with block
    scales, matching the layout used by ``ModelOptNvFp4LinearMethod``.

    Returns True if quantization was applied, False if skipped.
    """
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        convert_to_nvfp4_linear_kernel_format,
    )

    weight = layer.weight.data
    if weight.dtype not in (torch.bfloat16, torch.float16):
        logger.debug(
            "Skipping NVFP4 post-quant for %s: dtype is %s (not BF16/FP16)",
            name,
            weight.dtype,
        )
        return False

    device = weight.device
    output_size, input_size = weight.shape

    # Compute global scale: maps max(|weight|) → FP4_MAX
    weight_global_scale = torch.tensor(
        _FP4_MAX / weight.abs().max().clamp(min=1e-12).item(),
        dtype=torch.float32,
        device=device,
    )
    input_global_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    input_global_scale_inv = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Pack BF16 → FP4 uint8 + block scales
    weight_fp4, weight_scale = scaled_fp4_quant(
        weight, weight_global_scale, is_sf_swizzled_layout=False
    )

    # Install packed params on the layer (matching ModelOptNvFp4LinearMethod)
    del layer.weight
    layer.weight = Parameter(weight_fp4, requires_grad=False)
    layer.weight_scale = Parameter(
        weight_scale.to(torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_global_scale = Parameter(weight_global_scale, requires_grad=False)
    layer.input_global_scale = Parameter(input_global_scale, requires_grad=False)
    layer.input_global_scale_inv = Parameter(
        input_global_scale_inv, requires_grad=False
    )
    layer.alpha = Parameter(
        input_global_scale * weight_global_scale, requires_grad=False
    )
    layer.output_size_per_partition = output_size
    layer.input_size_per_partition = input_size

    # Convert to kernel-specific format
    convert_to_nvfp4_linear_kernel_format(backend, layer)

    # Replace quant_method with the NVFP4 adapter
    layer.quant_method = Fp4PostQuantLinearMethod(backend)

    logger.info(
        "Post-quantized %s to NVFP4: [%d, %d] BF16 (%.1f MB) → FP4 (%.1f MB)",
        name,
        output_size,
        input_size,
        output_size * input_size * 2 / 1e6,
        weight_fp4.numel() / 1e6,
    )
    return True


def apply_nvfp4_post_quant(model: nn.Module, layer_patterns: list[str]) -> int:
    """Post-quantize matching linear layers to NVFP4 on SM120+ GPUs.

    Walks model.named_modules() and converts any BF16/FP16 linear layer
    whose name ends with a suffix in ``layer_patterns`` to NVFP4.

    Only runs on SM120+ (Blackwell/GB10). No-op on older GPUs.

    Args:
        model: The model to quantize.
        layer_patterns: List of name suffixes to match (e.g., ["in_proj_qkvz"]).

    Returns:
        Number of layers converted.
    """
    if not current_platform.has_device_capability(120):
        return 0

    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        select_nvfp4_linear_backend,
    )

    backend = select_nvfp4_linear_backend()
    count = 0

    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        if not any(name.endswith(pat) for pat in layer_patterns):
            continue
        if post_quantize_linear_to_nvfp4(module, name, backend):
            count += 1

    if count > 0:
        torch.cuda.empty_cache()
        logger.info(
            "apply_nvfp4_post_quant: converted %d layer(s) to NVFP4 "
            "(patterns=%s)",
            count,
            layer_patterns,
        )

    return count
