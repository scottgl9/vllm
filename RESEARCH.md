# GB10 Spark — Open PR Research

PRs tracked during development of the `gb10-spark` branch.

## Implemented in This Branch

| PR | Title | Commit |
|----|-------|--------|
| #35568 | Marlin SM121 capability check (`is_` → `has_`) | H |
| #32704 | TRITON_PTXAS_PATH auto-configure | G |
| #35693 | NVFP4 global scale +inf overflow | Extra |
| #35356 | UMA memory reporting (is_integrated) | Already in HEAD |
| #34138 | MiniMax-M2 model support | Already in HEAD |
| #34822 | `is_blackwell_class()` + Blackwell attention backend priorities | N1 |
| #35576 | MLA weight access crash for NVFP4/INT4 | N2 |
| #34577 | NVFP4 weight scale BF16 underflow (marlin_utils_fp4) | N3 |

## Critical Open PRs (Not Yet Implemented)

| PR | Title | Status | Notes |
|----|-------|--------|-------|
| #35591 | ModelOpt KV cache dtype resolution | Open | KV cache config mismatch |
| #35041 | MTP NVFP4 weight shape mismatch | Open | `eh_proj` needs `ReplicatedLinear` |
| #35675 | Qwen3.5 NVFP4 MTP weight shape | Open | Same root cause as #35041 |

## GB10/SM121 Platform PRs

| PR | Title | Status | Notes |
|----|-------|--------|-------|
| #31740 | Comprehensive GB10 support | Open | 6-commit PR, overlaps with our changes |
| #31607 | SM 12.1 V1 engine support | Open | Early SM121 compat PR |

## NVFP4 Feature PRs

| PR | Title | Status |
|----|-------|--------|
| #35660 | NVFP4-quantized lm_head/embed_tokens | Open |
| #35733 | NVFP4 dense models on AMD/Hopper via emulation | Open |
| #35737 | NVFP4 MoE models on AMD/Hopper via emulation | Open |
| #34421 | LoRA with NVFP4 MoE models | Open |
| #34646 | EPLB + NVFP4 activation scales fix | Open |
| #32957 | RMSNorm NVFP4 quant operator | Open |

## MiniMax-Specific PRs

| PR | Title | Notes |
|----|-------|-------|
| #33303 | MiniMax-M2 PP+DP parallelism | Cherry-pick candidate |
| #33149 | MiniMax-M2 tool call parser | Cherry-pick candidate |

## Qwen3/3.5 PRs

| PR | Title | Notes |
|----|-------|-------|
| #34919 | Qwen3-Coder tool parser fix | Cherry-pick candidate |

## Other Blackwell PRs (Monitor)

| PR | Title |
|----|-------|
| #31089 | MXFP4 Triton on SM120 |
| #35360 | SymmMemCommunicator SM12.0 fix |
| #32930 | FusedMoE configs for RTX PRO 6000 |
| #34940 | Remove DBO xfail on Blackwell |
