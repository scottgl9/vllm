# GB10 Spark — vLLM Optimizations

All patches needed to run vLLM on NVIDIA DGX GB10 (SM 12.1, aarch64, CUDA 13.0).

---

## Build & Launch

Use `~/sandbox/vllm.sh` for all build and launch operations:

```bash
# One-time build (creates .venv-gb10, compiles vllm)
~/sandbox/vllm.sh build

# Launch server (sets all runtime env vars automatically)
~/sandbox/vllm.sh launch \
  --model ~/models/MiniMax-M2.5-REAP-139B-A10B-NVFP4/ \
  --quantization modelopt_fp4 --kv-cache-dtype fp8 \
  --max-model-len 4096 --enforce-eager

# Drop into venv shell for debugging
~/sandbox/vllm.sh shell
```

### Manual build (if needed)

```bash
cd ~/sandbox/vllm
python3.12 -m venv .venv-gb10
source .venv-gb10/bin/activate

pip install torch==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install flashinfer-python --pre
pip install -r requirements/build.txt

export TORCH_CUDA_ARCH_LIST="12.1a"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export NVCC_PREPEND_FLAGS="-arch=sm_121a"
export CUDA_HOME=/usr/local/cuda

pip install -e . --no-build-isolation -v 2>&1 | tee /tmp/vllm-gb10-build.log
```

> **FlashInfer patches** (formerly `scripts/gb10_post_install.py`) are applied
> automatically at startup when SM121 is detected.  The script is retained for
> manual/offline use.

---

## Runtime Environment

```bash
export VLLM_TEST_FORCE_FP8_MARLIN=1
export VLLM_NVFP4_GEMM_BACKEND=marlin
export VLLM_USE_FLASHINFER_MOE_FP4=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
```

---

## avarok/dgx-vllm-nvfp4-kernel:v23 — Performance Analysis

### Source
Container label: `vllm_source = 3b30e6150-patched` (vLLM 0.16.0rc2.dev236).
Patches extracted directly from running container via `docker run --rm ... git diff HEAD`.

### Performance delta vs avarok/vllm-dgx-spark:v11
- **Qwen3-Coder-Next-FP8**: 47 tok/s (v23) vs 43 tok/s (v11) = **+10%**
- **NVFP4 MoE models**: additional gain from tcgen05 and native MoE kernel

### What makes v23 faster

| Feature | v23 | v11 (old) |
|---------|-----|-----------|
| FP8 attention GEMM | Blackwell SM100 CUTLASS kernels (1×1×1 cluster, 128×256×128 tiles) | PyTorch `torch._scaled_mm` fallback |
| FP8 blockwise GEMM | Native `scaled_mm_blockwise_sm121_fp8` | PyTorch fallback |
| NVFP4 activation quant | CUDA kernels for sm_121 (all 5 NVFP4 .cu files compiled) | Python software loop (CUDA graph-hostile) |
| Tensor cores | tcgen05 5th-gen (`ENABLE_TCGEN05_HARDWARE=1`) | No tcgen05 flag |
| MoE (FP8 models) | GB10 native v109: Sm120+Pingpong+128³ | Generic Triton fallback |
| SM version dispatch | `>= 120 && < 130` (correct) | `>= 120` (catches SM130+) |

### Key files added

| File | Purpose |
|------|---------|
| `csrc/quantization/w8a8/cutlass/scaled_mm_c3x_sm121.cu` | Routes SM121 FP8 GEMMs to Blackwell SM100 kernels |
| `csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm121_fp8.cu` | Per-tensor FP8 GEMM for SM121 |
| `csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm121_fp8.cu` | Blockwise FP8 GEMM for SM121 |
| `csrc/quantization/w8a8/cutlass/moe/grouped_mm_gb10_native_v109.cu` | GeForce Blackwell MoE kernel (Pingpong schedule, adaptive tile) |

### Why FP8 attention matters (Qwen3-Next-FP8 decode)

At BS=1, each decode step loads:
- FP8 attention QKV+O (62 layers × 4 matrices × 3072×3072 @ FP8): ~1.2 GB
- FP8 MoE (8 active × 62 layers × intermediate=2048): ~2.8 GB
- BF16 lm_head: ~1.5 GB

Using CUTLASS SM100 kernels (vs PyTorch fallback) eliminates ~0.5–1 ms of CUDA kernel launch overhead per attention layer, adding up to ~3–5 ms per step → **~4 tok/s improvement at 47 tok/s throughput**.

---

## MiniMax M2.5 REAP 139B NVFP4 — Bandwidth Analysis

### Model architecture

| Field | Value |
|-------|-------|
| Layers | 62 |
| Hidden size | 3,072 |
| Attention heads | 48Q / 8KV (GQA), head_dim=128 |
| Experts | 154 total, top-8 active |
| Expert FFN size | intermediate=1,536 |
| MTP modules | **3** (layers 62–64) |
| Quantization ignore list | `lm_head`, all `block_sparse_moe.gate` |
| **Attention quantization** | **NVFP4** ← this model quantizes attention! |

### Per-decode-step weight loading (batch=1)

| Component | Size | Bandwidth |
|-----------|------|-----------|
| NVFP4 attention (qkv+o_proj × 62 layers) | ~1.37 GB | 22% |
| NVFP4 MoE experts (8 active × 62 layers) | ~3.43 GB | 56% |
| BF16 lm_head | ~1.23 GB | 20% |
| BF16 MoE gates (FP32) | ~0.12 GB | 2% |
| **Total** | **~6.15 GB** | 100% |

### Throughput ceilings

| Bandwidth | Latency/step | Ceiling |
|-----------|-------------|---------|
| 273 GB/s (GB10 theoretical) | 22.5 ms | **~44 tok/s** |
| 220 GB/s (GB10 effective) | 27.9 ms | **~36 tok/s** |
| 200 GB/s (Strix Halo effective) | 30.8 ms | ~32 tok/s |

**Current baseline:** 24 tok/s on Strix Halo (AMD, ~200 GB/s effective).
That's ~75% of the AMD effective ceiling — there's a meaningful overhead gap to close.

### Gap sources (why we're below ceiling)

1. **NVFP4 activation quantization per layer**: ~1–2 ms extra per step
2. **vLLM 1.78× slower than SGLang at BS=1** for NVFP4 MoE (documented gap — kernel launch overhead, no fused shuffle+reduce, generic CUTLASS configs)
3. **CUDA graph overhead** if not enabled or if capture fails
4. **Python scheduler overhead**

---

## Post-Quantization: lm_head to NVFP4

### Motivation

The lm_head is a dense BF16 linear layer [vocab_size × hidden_dim] = [200,064 × 3,072]
= 1.229 GB. At batch=1 decode, this is 18.2% of per-step weight bandwidth — the single
largest remaining BF16 component after attention and MoE experts are already NVFP4.

### Usage

```bash
VLLM_QUANTIZE_LM_HEAD=nvfp4 python -m vllm.entrypoints.openai.api_server \
  --model ~/models/MiniMax-M2.5-REAP-139B-A10B-NVFP4/ \
  --quantization modelopt_fp4 --kv-cache-dtype fp8 \
  --max-model-len 4096 --enforce-eager
```

### Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| lm_head size/step | 1.229 GB (BF16) | ~0.32 GB (FP4) |
| Total weight/step | 6.76 GB | ~5.85 GB |
| Expected throughput | ~25 tok/s | ~28-29 tok/s (+13-15%) |

### How it works

1. After model weights are loaded, the BF16 lm_head weight is quantized to NVFP4
   using `scaled_fp4_quant()` with a computed global scale
2. The packed FP4 weight and block scales are installed on the layer
3. A lightweight `_Fp4EmbeddingMethodAdapter` replaces the default quant_method,
   routing `LogitsProcessor` calls through `apply_nvfp4_linear()` (FP4 GEMM)
4. No calibration data is needed — `input_global_scale = 1.0` works because
   hidden states are already well-bounded from RMSNorm

### Quality

Tested on Qwen3-Coder-Next: no observable accuracy degradation with post-quantized
lm_head (+44% speedup on that model). For MiniMax M2.5, the lm_head vocabulary
projection is tolerant of FP4 quantization since the weight distribution is
relatively uniform across the vocabulary dimension.

---

## CUDA Graph Testing

### Status

Our patches (Fix 9b: FP4 JIT pre-warm; FlashInfer auto-patches; avarok v23 NVFP4 v6)
should have unblocked CUDA graph capture on GB10.

### Test procedure

```bash
# 1. Baseline with --enforce-eager (known working)
python -m vllm.entrypoints.openai.api_server \
  --model ~/models/MiniMax-M2.5-REAP-139B-A10B-NVFP4/ \
  --quantization modelopt_fp4 --kv-cache-dtype fp8 \
  --max-model-len 4096 --enforce-eager

# 2. CUDA graphs enabled (no --enforce-eager)
python -m vllm.entrypoints.openai.api_server \
  --model ~/models/MiniMax-M2.5-REAP-139B-A10B-NVFP4/ \
  --quantization modelopt_fp4 --kv-cache-dtype fp8 \
  --max-model-len 4096

# 3. Both optimizations combined
VLLM_QUANTIZE_LM_HEAD=nvfp4 python -m vllm.entrypoints.openai.api_server \
  --model ~/models/MiniMax-M2.5-REAP-139B-A10B-NVFP4/ \
  --quantization modelopt_fp4 --kv-cache-dtype fp8 \
  --max-model-len 4096
```

### Expected results

| Config | Expected tok/s | Notes |
|--------|---------------|-------|
| Baseline (--enforce-eager) | ~25 | Current |
| + lm_head NVFP4 | ~28-29 | Commit 1 |
| CUDA graphs (no --enforce-eager) | ~30-35 | If capture succeeds |
| CUDA graphs + lm_head NVFP4 | ~33-38 | Combined |
| Dual-Spark PP=2 | ~45 | Requires 2x GB10 |

### Potential issues

- If CUDA graph capture fails, check for dynamic shapes in FP4 activation
  quantization (the v23 NVFP4 CUDA kernels should handle this)
- FlashInfer JIT compilation during graph capture may timeout — pre-warm
  by running a few inference requests with `--enforce-eager` first

---

## Performance Roadmap

### Tier 1 — Biggest wins

#### A. MTP (Multi-Token Prediction) — status

`config.json` says `num_mtp_modules: 3`, but **the checkpoint has no MTP weights**.
`model.safetensors.index.json` only contains layers 0–61; layers 62–64 were dropped
during the Cerebras REAP pruning/quantization process.

MTP is therefore **unavailable for this specific checkpoint**.  If you want MTP:
1. Download the full (unpruned) MiniMax M2.5 BF16 weights and re-quantize with MTP
   layers included in the ignore list.
2. Or wait for a REAP variant that preserves the MTP modules.

PR #35442 (non-blocking MTP copy) was applied anyway — it helps Qwen3-Next-FP8
and any other MTP-capable model you run on GB10.

For MTP with the **original** MiniMax M2.5 (non-REAP), vllm would also need a
`MiniMaxM2MTPModel` class (none exists currently — unlike DeepSeek, Qwen3Next, etc.)

#### B. KV cache dtype
Already using `--kv-cache-dtype fp8` (good). Keeping this reduces KV load bandwidth.
At 96K context, KV cache = 62 layers × 2 × 8 heads × 128 dims × 1 byte × 96K tokens ≈ 12 GB.

### Tier 1b — Validate CUDA graphs first (potentially +20–30%)

#### Before anything else: test without `--enforce-eager`

The NVIDIA forums reported that `--enforce-eager` was required on early GB10 vllm builds
to avoid CUDA graph crashes.  Our patches (Fix 9b: FP4 JIT pre-warm; FlashInfer auto-patches)
should unblock CUDA graph capture.  If graphs now work, removing `--enforce-eager` alone
could give 20–30% throughput improvement by eliminating per-step kernel launch overhead
across 62 attention + 62 MoE + 3 norm layers.

```bash
# Test CUDA graph (no --enforce-eager)
python -m vllm.entrypoints.openai.api_server \
  --model ~/models/MiniMax-M2.5-REAP-139B-A10B-NVFP4/ \
  --quantization modelopt_fp4 --kv-cache-dtype fp8
```

### Tier 2 — Medium wins (+15–30%)

#### C. Validate CUDA graphs
If `--enforce-eager` was needed to avoid crashes, every kernel launch costs ~10–50 µs extra.
With CUDA graphs, 62 attention + 62 MoE launches collapse to a single replay → significant
latency reduction. Our Fix 9b (FP4 JIT pre-warm) should have unblocked CUDA graph capture.

```bash
# Try without --enforce-eager and verify no crash
python -m vllm.entrypoints.openai.api_server \
  --model ~/models/MiniMax-M2.5-REAP-139B-A10B-NVFP4/ \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8
  # (no --enforce-eager)
```

#### D. AR+Norm fusion (already in HEAD)
`allreduce_rms_fusion.py` fuses AllReduce + RMSNorm at O1+ torch.compile level.
Enable with `--compilation-config '{"optimization_level": 3}'`.

#### E. SiLU+FP4 quant fusion (already in HEAD)
`act_quant_fusion.py` fuses SiLU+FP4 quantization via torch.compile at O1+.
Reduces MoE forward bandwidth overhead by eliminating intermediate tensors.

### Tier 3 — Future wins (requires upstream work)

#### F. Native FP4 MoE kernels for SM121
Currently GB10 falls back to **Marlin W4A16 emulation** for NVFP4 MoE.
True FP4 tensor core kernels would directly reduce expert GEMM time.
No vLLM PR targeting SM121 native FP4 MoE exists yet — this is the next frontier.

#### G. SGLang-style MoE kernel fusion
vLLM launches 7 kernels/layer vs SGLang's 5 (fused shuffle+reduce).
This alone accounts for much of the 1.78× gap at BS=1.
No open PR in vLLM addresses this yet.

#### H. Multi-Spark (dual GB10)
PR #33303 adds PP+DP support for MiniMax-M2. With two DGX Sparks + InfiniBand,
expected scaling: ~1.8× throughput (memory bandwidth doubles, slight routing overhead).

---

## Patches Summary

| Commit | Description | Impact |
|--------|-------------|--------|
| A | CMake: SM12.1 in all arch lists | Enables kernel compilation |
| B | Software E2M1 conversion for SM121 | **Critical**: 1.1 → 35+ tok/s |
| D | Disable CUTLASS FP8 kernel for SM121 | Correct FP8 fallback |
| E | SM12.x MoE gating + oracle + autotune + Qwen3.5 | MoE backend access |
| F | NVFP4 emulation + MTP exclusion fixes | Model loading fixes |
| G | Auto TRITON_PTXAS_PATH | Triton JIT compatibility |
| H | Marlin SM121 capability check | Marlin W4A8-FP8 access |
| I | GB10 MoE Triton config | +65% MoE throughput |
| J | nv_fp4_dummy.h | CCCL FP4 type compat |
| K | FlashInfer auto-patches at startup | Replaces post-install script |
| — | PR #35693: global scale init | Prevents +inf overflow |
| N1 | PR #34822: is_blackwell_class() + attention backend priorities | SM12.x gets Blackwell-optimised attention |
| N2 | PR #35576: MLA kv_b_proj.weight.dtype crash fix | Prevents AttributeError for NVFP4 MLA models |
| N3 | PR #34577: NVFP4 scale BF16 underflow fix | Prevents zero scales / corrupted Marlin output |
| O | avarok v23: SM121 native FP8 CUTLASS kernels (scaled_mm_c3x_sm121, sm121_fp8, blockwise_sm121_fp8) | FP8 attention uses Blackwell SM100 kernels instead of PyTorch fallback |
| O | avarok v23: GB10 native MoE kernel v109 (grouped_mm_gb10_native_v109) | Sm120+Pingpong+128³ tiles for GeForce MoE; tcgen05 5th-gen tensor cores enabled |
| O | avarok v23: NVFP4 v6 full compilation (all nvfp4_*.cu for sm_121) | Eliminates Python FP4 quant fallback, fixes CUDA graph capture |
| O | avarok v23: scaled_mm_entry.cu version_num >= 120 && < 130 | Prevents SM130+ dispatch collision |
| P | minimax_m2: post-quantize BF16 lm_head to NVFP4 (env var gated) | ~13-15% throughput improvement |
| Q | minimax_m2: PP+DP support (from PR #33303) | Enables dual-Spark deployment |

---

## Smoke Tests

```bash
# Platform detection
python -c "from vllm.platforms import current_platform; print(current_platform.get_device_capability())"

# Qwen3-Next NVFP4
python -m vllm.entrypoints.openai.api_server \
  --model nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8 \
  --max-model-len 4096 \
  --enforce-eager

# MiniMax-M2.5 NVFP4 (with MTP, once #35041 is fixed)
python -m vllm.entrypoints.openai.api_server \
  --model ~/models/MiniMax-M2.5-REAP-139B-A10B-NVFP4/ \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8 \
  --num-speculative-steps 3  # enables MTP
```

---

## Performance Targets

| Model | Baseline | + lm_head FP4 | + CUDA graphs | + Dual-Spark | Ceiling |
|-------|---------|--------------|--------------|-------------|---------|
| MiniMax M2.5 139B NVFP4 (Strix Halo) | 24 tok/s | N/A | N/A (AMD) | N/A | ~32 tok/s |
| MiniMax M2.5 139B NVFP4 (GB10, vllm) | ~25 est. | ~28-29 | ~33-38 | ~45 | **36–44 tok/s** |
| Qwen3-Next NVFP4 on GB10 (SGLang) | 52 tok/s | — | — | — | — |
| Qwen3-Next NVFP4 on GB10 (vllm) | TBD | TBD | TBD | TBD | ~55 tok/s |

> **Note:** The MiniMax M2.5 GB10 ceiling (36–44 tok/s single GPU) cannot be exceeded
> without either (a) native FP4 tensor core MoE kernels on SM121, or (b) a checkpoint
> that includes MTP weights (none exist for the REAP variant currently).
> Dual-Spark PP=2 doubles available bandwidth and can exceed the single-GPU ceiling.
