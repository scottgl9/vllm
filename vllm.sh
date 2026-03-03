#!/bin/bash
# ~/sandbox/vllm.sh — Build and launch vLLM on DGX GB10 (SM 12.1 / CUDA 13.0)
#
# ── Commands ──────────────────────────────────────────────────────────────────
#
#   build    Create .venv-gb10 inside ~/sandbox/vllm and compile vllm. Run once
#            (or after pulling new commits).
#
#   launch   Activate the venv, set all runtime env vars, and exec the
#            OpenAI-compatible API server. All extra args are forwarded to
#            vllm.entrypoints.openai.api_server.
#
#   shell    Drop into a bash shell with the venv already activated and all
#            runtime env vars set — useful for one-off python invocations or
#            debugging.
#
# ── Build ─────────────────────────────────────────────────────────────────────
#
#   ~/sandbox/vllm.sh build
#
#   To force a clean rebuild, delete the venv first:
#     rm -rf ~/sandbox/vllm/.venv-gb10 && ~/sandbox/vllm.sh build
#
# ── Launch examples ───────────────────────────────────────────────────────────
#
#   MINIMAX_CACHE="$HOME/.cache/huggingface/hub/models--saricles--MiniMax-M2.5-REAP-139B-A10B-NVFP4-GB10/snapshots/bfdccfb01a260ccbbb93581600ad1c65ac0dfea0"
#
#   # 1. MiniMax M2.5 REAP 139B (saricles GB10 build) — baseline (eager, ~24 tok/s)
#   ~/sandbox/vllm.sh launch \
#     --model "$MINIMAX_CACHE" \
#     --quantization compressed-tensors \
#     --kv-cache-dtype fp8 \
#     --max-model-len 4096 \
#     --enforce-eager \
#     --trust-remote-code
#
#   # 2. MiniMax modelopt_fp4 from USB (alternate quant)
#   ~/sandbox/vllm.sh launch \
#     --model /mnt/usb/huggingface/MiniMax-M2.5-REAP-139B-A10B-NVFP4/ \
#     --quantization modelopt_fp4 \
#     --kv-cache-dtype fp8 \
#     --max-model-len 4096 \
#     --enforce-eager \
#     --trust-remote-code
#
#   # 3. + lm_head NVFP4 post-quantization (~28-29 tok/s, +13-15%)
#   VLLM_QUANTIZE_LM_HEAD=nvfp4 ~/sandbox/vllm.sh launch \
#     --model /mnt/usb/huggingface/MiniMax-M2.5-REAP-139B-A10B-NVFP4/ \
#     --quantization modelopt_fp4 \
#     --kv-cache-dtype fp8 \
#     --max-model-len 4096 \
#     --enforce-eager \
#     --trust-remote-code
#
#   # 4. + CUDA graphs (drop --enforce-eager, test for +20-30% more)
#   VLLM_QUANTIZE_LM_HEAD=nvfp4 ~/sandbox/vllm.sh launch \
#     --model /mnt/usb/huggingface/MiniMax-M2.5-REAP-139B-A10B-NVFP4/ \
#     --quantization modelopt_fp4 \
#     --kv-cache-dtype fp8 \
#     --max-model-len 4096 \
#     --trust-remote-code
#
#   # 5. Dual-Spark PP=2 (two GB10s connected via InfiniBand)
#   ~/sandbox/vllm.sh launch \
#     --model "$MINIMAX_CACHE" \
#     --quantization compressed-tensors \
#     --kv-cache-dtype fp8 \
#     --max-model-len 4096 \
#     --pipeline-parallel-size 2 \
#     --trust-remote-code
#
#   # 6. Qwen3.5-122B-A10B-NVFP4 (MoE, speculative decoding via qwen3_next_mtp)
#   ~/sandbox/vllm.sh Qwen3.5-NVFP4
#   # Override model path:  QWEN35_MODEL=/path/to/snapshot ~/sandbox/vllm.sh Qwen3.5-NVFP4
#
#   # 7. Qwen3-Coder-Next-FP8 (dense FP8, chunked-prefill, prefix-caching)
#   ~/sandbox/vllm.sh Qwen3-Coder-Next-FP8
#   # Override model path:  QWEN3_CODER_MODEL=Qwen/Qwen3-Coder-Next-FP8 ~/sandbox/vllm.sh Qwen3-Coder-Next-FP8
#
#   # 8. MiniMax M2.5 REAP 139B NVFP4 (modelopt_fp4 + lm_head NVFP4)
#   ~/sandbox/vllm.sh minimax
#   # Override model path:  MINIMAX_MODEL=/path/to/model ~/sandbox/vllm.sh minimax
#
# ── Environment variables ─────────────────────────────────────────────────────
#
# The following are set automatically by `launch` and `shell`.
# Override any of them by setting them before calling the script.
#
#   VLLM_QUANTIZE_LM_HEAD=nvfp4   Post-quantize BF16 lm_head to NVFP4 at load
#                                  time. Measured on MiniMax M2.5 (saricles GB10):
#                                    lm_head: 1229 MB BF16 → 307 MB FP4 (~4x)
#                                    throughput: 23.7 → 27.3 tok/s (+15%)
#                                  Default: unset (no change).
#
#   VLLM_NVFP4_GEMM_BACKEND       FP4 GEMM backend. Default: marlin.
#                                  Options: marlin | cutlass | flashinfer-cutlass
#                                  Benchmarked on GB10 (MiniMax M2.5, batch=1):
#                                    marlin:  ~23.7 tok/s (fastest)
#                                    cutlass: ~23.0 tok/s (-3%)
#
#   VLLM_MARLIN_USE_ATOMIC_ADD    Marlin atomic-add reduction (GB10 perf).
#                                  Default: 1 (enabled).
#
#   VLLM_TEST_FORCE_FP8_MARLIN    Force Marlin for FP8 GEMMs on GB10.
#                                  Default: 1.
#
#   VLLM_USE_FLASHINFER_MOE_FP4   Use FlashInfer for MoE FP4. Default: 0
#                                  (Marlin is used instead on GB10).
#
#   VLLM_USE_DEEP_GEMM            Enable DeepGEMM. Default: 0 (not SM121).
#
#   SAFETENSORS_FAST_GPU           GPU pinned-memory weight loading. Default: 1.
#                                  Avoids CPU double-copy on GB10 unified memory.
#
#   CUDA_CACHE_PATH                CUDA JIT kernel cache. Default: ~/.nv/...
#   CUDA_CACHE_MAXSIZE             Cache size limit in bytes. Default: 4 GB.
#
# ── Slow model loading on DGX Spark ───────────────────────────────────────────
#
# Default safetensors mmap loading is pathologically slow on GB10 due to the
# ARM SMMU/IOMMU synchronizing every page between Grace CPU and Blackwell GPU.
# Fix options in priority order:
#
#   1. --load-format fastsafetensors   (pip install "fastsafetensors>=0.1.10")
#      Bypasses CPU staging; copies tensors directly to GPU. Reported 7.5min→1min
#      on DGX Spark. May have issues with FP4/FP8 dtypes (test first).
#
#   2. --safetensors-load-strategy eager
#      Reads full shards to CPU RAM before GPU transfer; avoids mmap page faults.
#      ~6x faster than default mmap (benchmark: 8m41s→1m28s on 80B model).
#      Safe fallback if fastsafetensors has dtype incompatibilities.
#
#   3. NVMe read-ahead (one-time system tuning, ~2x on mmap path):
#        sudo bash -c "echo 8192 > /sys/block/nvme0n1/queue/read_ahead_kb"
#      (check device name with: lsblk)
#
#   4. SAFETENSORS_FAST_GPU=1 — already set above; avoids pageable H2D copies.
#
#   5. sharded_state save/load — save once after Marlin repack, reload instantly:
#        # One-time save:
#        python -c "
#        from vllm import LLM
#        llm = LLM(model='<path>', quantization='compressed-tensors')
#        llm.llm_engine.model_executor.save_sharded_state(path='/path/to/cache/')
#        "
#        # Fast reload:
#        ./vllm.sh launch --model /path/to/cache/ --load-format sharded_state ...
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(pwd)"
VENV_DIR="${VLLM_DIR}/.venv-gb10"
PYTHON="python3.12"

# ── Build-time env vars ───────────────────────────────────────────────────────
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="12.1a"
# Do NOT set -arch here: cmake handles arch via TORCH_CUDA_ARCH_LIST.
# Adding -arch=sm_121a here conflicts with cmake's own sm_121 gencode flags
# (nvcc fatal: same GPU code for non family-specific and family-specific arch).
export NVCC_PREPEND_FLAGS="-DENABLE_TCGEN05_HARDWARE=1"
export TRITON_PTXAS_PATH="${CUDA_HOME}/bin/ptxas"
# Parallelize nvcc and cmake builds. 8 is a good default for GB10 (64-core ARM).
# Set MAX_JOBS before calling this script to override.
export MAX_JOBS="${MAX_JOBS:-8}"
export CUDA_NVCC_FLAGS="${CUDA_NVCC_FLAGS:---threads 4}"

# ── Runtime env vars (applied for `launch`) ───────────────────────────────────
setup_runtime_env() {
    # CUDA / Triton
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export TRITON_PTXAS_PATH="${CUDA_HOME}/bin/ptxas"

    # CUDA kernel cache — persists JIT-compiled kernels across restarts
    export CUDA_CACHE_PATH="${HOME}/.nv/ComputeCache"
    export CUDA_CACHE_MAXSIZE=4294967296   # 4 GB

    # FlashInfer + inductor cache dirs
    export FLASHINFER_WORKSPACE_DIR="${HOME}/.cache/flashinfer"
    export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torch/inductor"

    # Faster safetensors weight loading via GPU pinned memory
    export SAFETENSORS_FAST_GPU=1

    # torch.compile / inductor thread count (avoids saturating CPUs during JIT)
    export TORCH_COMPILE_THREADS=4
    export TORCHINDUCTOR_COMPILE_THREADS=4

    # vLLM: NVFP4 GEMM via Marlin (fastest path on SM121). Override by setting
    # VLLM_NVFP4_GEMM_BACKEND before calling this script (e.g. vllm_cutlass).
    export VLLM_NVFP4_GEMM_BACKEND="${VLLM_NVFP4_GEMM_BACKEND:-marlin}"
    # vLLM: Marlin atomic-add reduction (improves throughput on GB10)
    export VLLM_MARLIN_USE_ATOMIC_ADD=1
    # vLLM: force Marlin path for FP8 GEMMs (CUTLASS FP8 path not SM121-ready)
    export VLLM_TEST_FORCE_FP8_MARLIN=1
    # vLLM: disable FlashInfer MoE FP4 (Marlin path is used instead)
    export VLLM_USE_FLASHINFER_MOE_FP4=0
    # vLLM: disable DeepGEMM (not supported on SM121)
    export VLLM_USE_DEEP_GEMM=0
    # vLLM: spawn method avoids fork-related CUDA issues with multiprocessing
    export VLLM_WORKER_MULTIPROC_METHOD=spawn

    # PyTorch allocator: expandable segments avoids fragmentation on large models
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}

# ── Helpers ───────────────────────────────────────────────────────────────────
info()    { echo -e "\033[1;34m[vllm]\033[0m $*"; }
success() { echo -e "\033[1;32m[vllm]\033[0m $*"; }
warn()    { echo -e "\033[1;33m[vllm]\033[0m $*"; }
die()     { echo -e "\033[1;31m[vllm]\033[0m ERROR: $*" >&2; exit 1; }

# ── Subcommands ───────────────────────────────────────────────────────────────

cmd_build() {
    info "Building vLLM on GB10 (SM 12.1, CUDA 13.0)"
    info "  Source : ${VLLM_DIR}"
    info "  Venv   : ${VENV_DIR}"
    echo ""

    [[ -d "${VLLM_DIR}" ]] || die "vllm source not found at ${VLLM_DIR}"

    # Verify CUDA is present before spending time building
    [[ -x "${CUDA_HOME}/bin/nvcc" ]] || die "nvcc not found at ${CUDA_HOME}/bin/nvcc. Is CUDA installed?"

    # Create venv if it doesn't exist
    if [[ ! -d "${VENV_DIR}" ]]; then
        info "Creating venv with ${PYTHON}..."
        "${PYTHON}" -m venv "${VENV_DIR}"
    else
        warn "Venv already exists — reusing (delete ${VENV_DIR} to rebuild from scratch)"
    fi

    # Activate
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    # Pin the pytorch whl index for ALL subsequent pip calls in this shell.
    # Without this, `pip install -r requirements/build.txt` (which has a bare
    # `torch==2.10.0` with no local suffix) re-resolves from PyPI and pulls
    # the CPU wheel, overwriting the cu130 build we just installed.
    export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu130"

    info "Upgrading pip..."
    pip install -q --upgrade pip

    # ── Python packages ─────────────────────────────────────────────────────

    info "Installing PyTorch 2.10 for CUDA 13.0 (cu130, aarch64)..."
    pip install "torch==2.10.0+cu130" \
        --index-url https://download.pytorch.org/whl/cu130

    info "Installing Triton >=3.3.0 with SM121a ptxas support..."
    # triton <3.3.0 produces a fatal ptxas error on sm_121a; pip will pick the
    # latest compatible version from PyPI (triton is published there, not on
    # the pytorch whl index).
    pip install "triton>=3.3.0"

    info "Installing FlashInfer (pre-release, for SM120/121 NVFP4 kernels)..."
    pip install flashinfer-python --pre

    info "Installing vLLM build requirements..."
    # PIP_EXTRA_INDEX_URL is set above so torch==2.10.0 in build.txt resolves
    # to the cu130 variant, not the CPU one from PyPI.
    pip install -r "${VLLM_DIR}/requirements/build.txt"

    # requirements/build.txt has bare `torch==2.10.0` which pip may resolve to
    # the CPU wheel from PyPI (pip treats 2.10.0+cu130 as a different version
    # than 2.10.0 for exact == matching). Force-reinstall cu130 BEFORE cmake
    # runs so that torch.version.cuda is set and CUDA extensions are compiled.
    info "Re-pinning torch+cu130 before CUDA compilation..."
    pip install --force-reinstall --no-deps "torch==2.10.0+cu130" \
        --index-url https://download.pytorch.org/whl/cu130

    # ── Compile vLLM C++/CUDA extensions ────────────────────────────────────
    # Source file modifications on gb10-spark-minimax branch:
    #
    # CMakeLists.txt:
    #   - scaled_mm_blockwise_sm121_fp8.cu: DISABLED (CUTLASS __CUTLASS_UNUSED
    #     macro fails under CUDA 13.0; stub throws "not implemented" at runtime)
    #   - grouped_mm_gb10_native_v109.cu:   DISABLED (references missing
    #     vllm::ScaledEpilogueBiasForward; SM120 MoE kernel handles GB10 instead)
    #   - GB10 NVFP4 v6 cmake section: REPLACED with status-only message
    #     (standard FP4 section already covers SM121 via cuda_archs_loose_intersection)
    #
    # scaled_mm_entry.cu: SM121-specific dispatch added BEFORE the SM120 branch
    #   so GB10 routes to SM121 kernels (compiled sm_121f + ENABLE_TCGEN05_HARDWARE)
    #   instead of the generic SM120 kernels (sm_120f).
    #
    # scaled_mm_c3x_sm121.cu: FIXED to call cutlass_scaled_mm_sm121_fp8()
    #   (from scaled_mm_sm121_fp8.cu, compiled for sm_121f) instead of the
    #   SM100 functions which aren't compiled for SM121-only builds.
    #
    # scaled_mm_helper.hpp: Added if constexpr guard for nullptr blockwise_func
    #   (same pattern as Int8Func) so SM121 can pass nullptr for blockwise.
    #
    # vllm_flash_attn/__init__.py: PATCHED to warn (not raise ImportError) when
    # FA2/FA3 extensions are unavailable — FlashInfer is used as fallback.
    #
    # setup.py: FA2/FA3 CMakeExtension marked optional=True so build succeeds
    # even if flash-attention doesn't compile on CUDA 13.0.

    info "Compiling vLLM (editable install, no-build-isolation)..."
    info "  TORCH_CUDA_ARCH_LIST = ${TORCH_CUDA_ARCH_LIST}"
    info "  NVCC_PREPEND_FLAGS   = ${NVCC_PREPEND_FLAGS}"
    info "  MAX_JOBS             = ${MAX_JOBS}"
    pushd "${VLLM_DIR}" > /dev/null
    pip install -e . --no-build-isolation -v 2>&1 | tee /tmp/vllm-gb10-build.log
    popd > /dev/null

    # ── Force-pin torch+cu130 last ───────────────────────────────────────────
    # `pip install -e .` pulls torchvision/torchaudio which can drag in the
    # CPU torch as a transitive dep.  Reinstall to guarantee the CUDA build.
    info "Pinning torch+cu130 (force-reinstall to survive transitive deps)..."
    pip install --force-reinstall --no-deps "torch==2.10.0+cu130" \
        --index-url https://download.pytorch.org/whl/cu130

    # ── Verify the build ─────────────────────────────────────────────────────

    info "Verifying installation..."
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda or 'None')")
    VLLM_VER=$(python -c "import vllm; print(getattr(vllm, '__version__', 'ok'))" 2>/dev/null || echo "ok")
    PLATFORM=$(python -c "from vllm.platforms import current_platform; print(current_platform.get_device_capability())" 2>/dev/null || echo "(GPU not visible in this shell)")

    echo ""
    echo "  torch           : ${TORCH_VER}"
    echo "  torch.version.cuda : ${TORCH_CUDA}"
    echo "  vllm            : ${VLLM_VER}"
    echo "  platform cap    : ${PLATFORM}"
    echo ""

    if [[ "${TORCH_CUDA}" == "None" || "${TORCH_CUDA}" == "" ]]; then
        warn "torch.version.cuda is None — CPU-only torch was installed."
        warn "This usually means the pytorch whl index did not serve a cu130"
        warn "wheel for this Python/platform combo.  Check manually:"
        warn "  source ${VENV_DIR}/bin/activate"
        warn "  pip install torch==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130"
    else
        success "Build complete. Log: /tmp/vllm-gb10-build.log"
    fi
    echo ""
    info "To launch:"
    echo "  ./vllm.sh launch --model <model_path> --quantization modelopt_fp4 \\"
    echo "      --kv-cache-dtype fp8 --max-model-len 4096"
}

cmd_launch() {
    [[ -d "${VENV_DIR}" ]] || die "Venv not found at ${VENV_DIR}. Run: ./vllm.sh build"

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    setup_runtime_env

    info "Launching vLLM OpenAI-compatible server"
    info "  VLLM_NVFP4_GEMM_BACKEND      = ${VLLM_NVFP4_GEMM_BACKEND}"
    info "  VLLM_MARLIN_USE_ATOMIC_ADD   = ${VLLM_MARLIN_USE_ATOMIC_ADD}"
    info "  VLLM_QUANTIZE_LM_HEAD        = ${VLLM_QUANTIZE_LM_HEAD:-<unset>}"
    info "  SAFETENSORS_FAST_GPU         = ${SAFETENSORS_FAST_GPU}"
    info "  CUDA_CACHE_PATH              = ${CUDA_CACHE_PATH}"
    info "  TRITON_PTXAS_PATH            = ${TRITON_PTXAS_PATH}"
    echo ""

    # cd away from ~/sandbox before exec: if CWD contains a directory named
    # 'vllm', Python adds it to sys.path and it shadows the editable install,
    # causing "unknown location" import errors.
    cd /tmp
    exec python -m vllm.entrypoints.openai.api_server "$@"
}

cmd_qwen35_nvfp4() {
    # Auto-detect snapshot dir under the Sehyo HF cache, or use QWEN35_MODEL
    local base="${HOME}/.cache/huggingface/hub/models--Sehyo--Qwen3.5-122B-A10B-NVFP4/snapshots"
    local model="${QWEN35_MODEL:-}"
    if [[ -z "$model" ]]; then
        model=$(ls -td "${base}"/*/  2>/dev/null | head -1)
        model="${model%/}"
        [[ -n "$model" ]] || die "Qwen3.5-NVFP4 model not found at ${base}. Set QWEN35_MODEL=/path/to/snapshot"
    fi

    info "Preset: Qwen3.5-122B-A10B-NVFP4 (compressed-tensors, speculative qwen3_next_mtp)"
    info "  Model: ${model}"

    cmd_launch \
        --model "${model}" \
        --served-model-name qwen3-coder-next \
        --quantization compressed-tensors \
        --kv-cache-dtype fp8 \
        --gpu-memory-utilization 0.88 \
        --max-model-len 65536 \
        --max-num-seqs 3 \
        --attention-backend flashinfer \
        --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
        --no-enable-chunked-prefill \
        --served-model-name qwen3-coder-next \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --language-model-only \
        --trust-remote-code \
        "$@"
}

cmd_qwen3_coder_next_nvfp4() {
    local model="${QWEN3_CODER_NVFP4_MODEL:-GadflyII/Qwen3-Coder-Next-NVFP4}"

    info "Preset: Qwen3-Coder-Next-NVFP4 (GadflyII, compressed-tensors, chunked-prefill)"
    info "  Model: ${model}"

    cmd_launch \
        --model "${model}" \
        --quantization compressed-tensors \
        --served-model-name qwen3-coder-next \
        --host 0.0.0.0 \
        --port 8000 \
        --gpu-memory-utilization 0.8 \
        --kv-cache-dtype fp8_e4m3 \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --max-num-seqs 64 \
        --max-num-batched-tokens 8192 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --max-model-len 131072 \
        "$@"
}

cmd_qwen3_coder_next_fp8() {
    local model="${QWEN3_CODER_MODEL:-Qwen/Qwen3-Coder-Next-FP8}"

    info "Preset: Qwen3-Coder-Next-FP8 (chunked-prefill, prefix-caching)"
    info "  Model: ${model}"

    cmd_launch \
        --model "${model}" \
        --served-model-name qwen3-coder-next \
        --host 0.0.0.0 \
        --port 8000 \
        --gpu-memory-utilization 0.86 \
        --kv-cache-dtype fp8_e4m3 \
        --stream-interval 5 \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --max-num-seqs 4 \
        --max-num-batched-tokens 8192 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --max-model-len 131072 \
        "$@"
}

cmd_minimax() {
    local model="${MINIMAX_MODEL:-${HOME}/models/MiniMax-M2.5-REAP-139B-A10B-NVFP4}"
    export VLLM_QUANTIZE_LM_HEAD="${VLLM_QUANTIZE_LM_HEAD:-nvfp4}"

    info "Preset: MiniMax M2.5 REAP 139B NVFP4 (modelopt_fp4, lm_head=${VLLM_QUANTIZE_LM_HEAD})"
    info "  Model: ${model}"

    cmd_launch \
        --model "${model}" \
        --quantization modelopt_fp4 \
        --kv-cache-dtype fp8 \
        --max-model-len 4096 \
        --enforce-eager \
        "$@"
}

cmd_shell() {
    [[ -d "${VENV_DIR}" ]] || die "Venv not found at ${VENV_DIR}. Run: ./vllm.sh build"
    info "Activating vLLM venv — type 'deactivate' to exit"
    setup_runtime_env
    exec bash --rcfile <(echo "source '${VENV_DIR}/bin/activate'; PS1='(vllm-gb10) \u@\h:\w\$ '")
}

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [args]

Commands:
  build                     Create venv and compile vLLM from ~/sandbox/vllm
  launch [vllm args]        Start the OpenAI-compatible server (raw args)
  shell                     Drop into an activated venv shell

  Qwen3.5-NVFP4  [args]       Qwen3.5-122B-A10B-NVFP4 (MoE, compressed-tensors,
                               speculative qwen3_next_mtp, max-len 65536)
  Qwen3-Coder-Next-NVFP4 [args]
                               GadflyII/Qwen3-Coder-Next-NVFP4 (compressed-tensors,
                               chunked-prefill, prefix-caching, max-len 65536)
  Qwen3-Coder-Next-FP8 [args]
                               Qwen3-Coder-Next-FP8 (dense FP8, chunked-prefill,
                               prefix-caching, max-len 131072)
  minimax [args]               MiniMax M2.5 REAP 139B NVFP4 (modelopt_fp4,
                               lm_head NVFP4, enforce-eager, max-len 4096)

Environment overrides:
  QWEN35_MODEL=<path>              Override Qwen3.5-NVFP4 snapshot path
  QWEN3_CODER_NVFP4_MODEL=<path>  Override Qwen3-Coder-Next-NVFP4 model (default: HF hub)
  QWEN3_CODER_MODEL=<path>        Override Qwen3-Coder-Next-FP8 model (default: HF hub)
  MINIMAX_MODEL=<path>             Override MiniMax model path
  VLLM_QUANTIZE_LM_HEAD=nvfp4     Post-quantize lm_head to NVFP4 (~13-15% speedup)

Examples:
  # Named model presets (recommended)
  ./vllm.sh Qwen3.5-NVFP4
  ./vllm.sh Qwen3-Coder-Next-NVFP4
  ./vllm.sh Qwen3-Coder-Next-FP8
  ./vllm.sh minimax

  # Pass extra args to any preset (appended to preset defaults)
  ./vllm.sh minimax --max-model-len 8192
  QWEN35_MODEL=/path/to/snapshot ./vllm.sh Qwen3.5-NVFP4

  # Raw launch (full control)
  VLLM_QUANTIZE_LM_HEAD=nvfp4 ./vllm.sh launch \\
    --model ~/models/MiniMax-M2.5-REAP-139B-A10B-NVFP4/ \\
    --quantization modelopt_fp4 --kv-cache-dtype fp8 \\
    --max-model-len 4096 --enforce-eager

EOF
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
CMD="${1:-}"
shift || true

case "${CMD}" in
    build)   cmd_build ;;
    launch)  cmd_launch "$@" ;;
    shell)   cmd_shell ;;
    Qwen3.5-NVFP4|qwen3.5-nvfp4|qwen35-nvfp4) cmd_qwen35_nvfp4 "$@" ;;
    Qwen3-Coder-Next-NVFP4|qwen3-coder-next-nvfp4) cmd_qwen3_coder_next_nvfp4 "$@" ;;
    Qwen3-Coder-Next-FP8|qwen3-coder-next-fp8) cmd_qwen3_coder_next_fp8 "$@" ;;
    minimax|MiniMax) cmd_minimax "$@" ;;
    ""|help|-h|--help) usage ;;
    *) die "Unknown command: ${CMD}. Run './vllm.sh help' for usage." ;;
esac
