# GB10 Spark — vLLM Optimizations

All patches needed to run vLLM on NVIDIA DGX GB10 (SM 12.1, aarch64, CUDA 13.0).

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

> **Note:** FlashInfer header patches (formerly `scripts/gb10_post_install.py`)
> are now applied automatically at startup when SM121 is detected.  The script
> is retained for manual/offline use only.

## Runtime Environment

```bash
export VLLM_TEST_FORCE_FP8_MARLIN=1
export VLLM_NVFP4_GEMM_BACKEND=marlin
export VLLM_USE_FLASHINFER_MOE_FP4=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
```

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

# MiniMax-M2.5 NVFP4
python -m vllm.entrypoints.openai.api_server \
  --model ~/models/MiniMax-M2.5-REAP-139B-A10B-NVFP4/ \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8
```

## Performance Targets

| Model | Target |
|-------|--------|
| Qwen3-Next NVFP4 | ≥ 40 tok/s |
| Qwen3-Next NVFP4 + MTP | ≥ 60 tok/s |
| MiniMax-M2.5 NVFP4 | ≥ 25 tok/s |
