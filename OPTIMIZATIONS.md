# GB10 Spark — vLLM Optimizations

All patches needed to run vLLM on NVIDIA DGX GB10 (SM 12.1, aarch64, CUDA 13.0).

---

## Build

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

## Performance Roadmap

### Tier 1 — Biggest wins (up to 3× combined)

#### A. Enable MTP (Multi-Token Prediction)
The MiniMax M2.5 REAP model ships with **3 MTP modules** — lightweight speculative
decoder layers at indices 62–64 that predict 3 tokens ahead each step.

| Scenario | tok/s estimate |
|----------|---------------|
| Baseline (no MTP) | 24–30 |
| MTP × 2 (70% acceptance, 2 draft tokens) | 48–60 |
| MTP × 3 (70% acceptance, 3 draft tokens) | 57–75 |

**Blockers to fix first:**
- PR #35041 — weight shape mismatch when using MTP + NVFP4 (OPEN, must integrate)
- PR #35442 — 6ms CPU–GPU sync per MTP step → 200µs (OPEN, must integrate)

#### B. KV cache dtype
Already using `--kv-cache-dtype fp8` (good). Keeping this reduces KV load bandwidth.
At 96K context, KV cache = 62 layers × 2 × 8 heads × 128 dims × 1 byte × 96K tokens ≈ 12 GB.

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

| Model | Current | Target (no MTP) | Target (MTP) |
|-------|---------|-----------------|--------------|
| MiniMax M2.5 139B NVFP4 on Strix Halo | 24 tok/s | — | — |
| MiniMax M2.5 139B NVFP4 on GB10 | ~25 tok/s est. | 33–36 tok/s | 50–75 tok/s |
| Qwen3-Next NVFP4 on GB10 (SGLang) | 52 tok/s (sglang.mine) | — | — |
| Qwen3-Next NVFP4 on GB10 (vllm) | TBD | 40–48 tok/s | TBD |
