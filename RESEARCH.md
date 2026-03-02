# GB10 Spark — Open PR Research

PRs tracked during development of the `gb10-spark` branch.

---

## Implemented in This Branch

| PR / Source | Title | Commit |
|----|-------|--------|
| #35568 | Marlin SM121 capability check (`is_` → `has_`) | H |
| #32704 | TRITON_PTXAS_PATH auto-configure | G |
| #35693 | NVFP4 global scale +inf overflow | Extra |
| #35356 | UMA memory reporting (is_integrated) | Already in HEAD |
| #34138 | MiniMax-M2 model support | Already in HEAD |
| #34822 | `is_blackwell_class()` + Blackwell attention backend priorities | N1 |
| #35576 | MLA weight access crash for NVFP4/INT4 | N2 |
| #34577 | NVFP4 weight scale BF16 underflow (marlin_utils_fp4) | N3 |
| avarok v23 | SM121 native FP8 scaled_mm CUTLASS kernels (5 files) | O |
| avarok v23 | GB10 native MoE kernel v109 (Pingpong+128³+tcgen05) | O |
| avarok v23 | NVFP4 v6: all fp4 kernels compiled for sm_121 | O |
| avarok v23 | scaled_mm_entry.cu dispatch fix (>= 120 && < 130) | O |
| avarok v23 | ENABLE_TCGEN05_HARDWARE=1 for SM12.x | O |

### avarok Container Source Analysis

`avarok/dgx-vllm-nvfp4-kernel:v23` (Feb 2026) is the high-performance container.
- Image label: `vllm_source = 3b30e6150-patched` (vLLM 0.16.0rc2.dev236)
- Docker Hub: `avarok/dgx-vllm-nvfp4-kernel`
- GitHub: `Avarok-Cybersecurity/dgx-vllm`

Key env vars in v23:
```
ENABLE_TCGEN05_HARDWARE=1
NVCC_PREPEND_FLAGS=-arch=sm_121a -DENABLE_TCGEN05_HARDWARE=1
```

Performance: **47 tok/s** for Qwen3-Coder-Next-FP8 (vs 43 tok/s in `vllm-dgx-spark:v11`).

Root cause of 10% improvement: SM121 FP8 attention now uses Blackwell SM100 CUTLASS
kernels (128×256×128 tiles, 1×1×1 cluster, KernelScheduleAuto) instead of PyTorch
`torch._scaled_mm`. Plus `ENABLE_TCGEN05_HARDWARE=1` enables tcgen05 tensor core
instructions in compiled CUTLASS kernels.

## Already Merged in HEAD (no action needed)

| PR | Title | Merged |
|----|-------|--------|
| #33517 | SM121 CUTLASS `enable_sm120_or_later` for FP8 blockwise GEMM | 2026-02-07 |
| #33417 | FlashInfer NVFP4 MoE for SM120/121 (family-based check) | 2026-01-30 |
| #34424 | FP8 GEMM performance optimization for SM120 | 2026-02-25 |
| #34846 | Improve Triton FusedMoE defaults (used as fallback on GB10) | 2026-02-18 |
| #34718 | SiLU+FP4 quant fusion via torch.compile for O1+ | 2026-02-17 |
| #34899 | Bump FlashInfer + re-enable NVFP4 AllReduce+RMSNorm fusion | 2026-02-20 |
| #30885 | 8x4 SF tiling for NVFP4 small-batch decode (25–35% faster BS≤32) | 2026-01-13 |
| #32974 | Flash Attention 4 (FA4) integration | 2026-01-23 |
| #35422 | Extract KV cache update from FlashInfer forward (overlap with compute) | 2026-02-26 |
| #34673 | Fix MoE routing for `num_expert_group=None` (MiniMax-M2.1) | 2026-02-17 |

---

## Critical Open PRs — Performance Impact

### MTP (Multi-Token Prediction) — Biggest Lever

The MiniMax M2.5 REAP model contains **3 MTP modules** (layers 62, 63, 64).
If MTP is enabled and working, each decode step speculates 3 tokens ahead.
At ~70% acceptance rate, this yields **~2–2.5x throughput** (24 → 48–60 tok/s).

| PR | Title | Status | Impact |
|----|-------|--------|--------|
| **#35041** | MTP + NVFP4 weight shape mismatch fix | Open | **Required** for MTP+NVFP4 |
| **#35442** | Non-blocking MTP token copy (6ms → 200µs CPU-GPU sync) | Open | **+~6ms/step** with MTP |
| #32592 | MTP: delete redundant position-equals-zero op | Open | Minor |

### MiniMax-Specific

| PR | Title | Status | Impact |
|----|-------|--------|--------|
| **#33303** | MiniMax-M2 pipeline + data parallelism (PP + DP) | Open | Multi-Spark 2–4x |
| #33149 | Fix MiniMax tool call parser for `stream_interval > 1` | Open | Correctness |
| #34863 | Fix compressed-tensors FP8 FlashInfer scale propagation | Open | Qwen3 accuracy |

### GB10 / SM121 Remaining

| PR | Title | Status | Notes |
|----|-------|--------|-------|
| **#31740** | Comprehensive GB10 support (6 commits: CMake, Triton, MoE configs) | Open | Overlaps our branch; has extra Triton MoE tile configs for GB10 |
| #31607 | SM 12.1 V1 engine graceful fallback | Open | Safety net for CUTLASS ops |

---

## NVFP4 Feature PRs

| PR | Title | Status |
|----|-------|--------|
| #35660 | NVFP4-quantized lm_head/embed_tokens | Open |
| #35733 | NVFP4 dense models on AMD/Hopper via emulation | Open |
| #35737 | NVFP4 MoE models on AMD/Hopper via emulation | Open |
| #34421 | LoRA with NVFP4 MoE models | Open |
| #34646 | EPLB + NVFP4 activation scales fix | Open |
| #32957 | RMSNorm NVFP4 quant operator | Open |

---

## MiniMax-Specific PRs (Cherry-pick Candidates)

| PR | Title | Notes |
|----|-------|-------|
| #33303 | MiniMax-M2 PP+DP parallelism | Dual-Spark scaling |
| #33149 | MiniMax-M2 tool call parser | Streaming fix |

---

## Qwen3/3.5 PRs

| PR | Title | Notes |
|----|-------|-------|
| #34919 | Qwen3-Coder tool parser fix | Cherry-pick candidate |

---

## Other Blackwell PRs (Monitor)

| PR | Title |
|----|-------|
| #31089 | MXFP4 Triton on SM120 |
| #35360 | SymmMemCommunicator SM12.0 fix |
| #32930 | FusedMoE configs for RTX PRO 6000 |
| #34940 | Remove DBO xfail on Blackwell |
| #33540 | UCX MNNVL protocol for GB-series (dual-Spark NVLink) |

---

## Performance Findings: vLLM vs SGLang for NVFP4 MoE

The [HuggingFace TFLOPS Gap analysis](https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison)
quantifies vLLM's current disadvantage vs SGLang for NVFP4 MoE at BS=1 on B200:

- **BS=1 decode:** SGLang 206.9 µs/layer vs vLLM 369.5 µs/layer → **vLLM 1.78× slower**
- **BS=128 decode:** SGLang 0.433 ms/layer vs vLLM 0.604 ms/layer → **28% slower**

Root causes (no vLLM PR addresses these yet):
1. vLLM launches 7 kernels per MoE layer vs SGLang's 5 (shuffle+reduce not fused)
2. vLLM uses generic CUTLASS 3.x configs vs SGLang's `KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100`
3. No adaptive grid sizing for small batches in vLLM

This gap explains why `sglang.mine` achieves 52 tok/s on Qwen3-Coder-Next on GB10 while
vLLM gets ~24 tok/s on MiniMax M2.5 (different model, but the kernel efficiency gap is real).
