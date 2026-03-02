# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GB10 (SM121) FlashInfer compatibility patches.

SM121 (NVIDIA DGX GB10 Spark) supports FP4 tensor core MMA but lacks the
``cvt.rn.satfinite.e2m1x2.f32`` PTX instruction for hardware float-to-E2M1
conversion.  FlashInfer's JIT-compiled kernels reference this instruction via
CUTLASS and TRT-LLM headers, so they must be patched before the first JIT
compilation on SM121.

This module is called automatically from ``vllm.utils.flashinfer`` when
FlashInfer is first accessed on an SM121 device.  All patches are idempotent;
a sentinel comment is used to detect prior application.
"""

import os
import shutil
import site
from functools import cache

from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# FlashInfer data-directory discovery
# ---------------------------------------------------------------------------


@cache
def _find_flashinfer_data_dir() -> str | None:
    """Return the FlashInfer data directory, or None if not found."""
    candidates: list[str] = []
    try:
        candidates += [
            os.path.join(sp, "flashinfer", "data")
            for sp in site.getsitepackages()
        ]
        candidates.append(
            os.path.join(site.getusersitepackages(), "flashinfer", "data")
        )
    except Exception:
        pass

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        for pyver in ("python3.12", "python3.11", "python3.13", "python3.10"):
            candidates.append(
                os.path.join(
                    venv, "lib", pyver, "site-packages", "flashinfer", "data"
                )
            )

    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


# ---------------------------------------------------------------------------
# Individual patch functions
# ---------------------------------------------------------------------------


def _patch_float_subbyte(data_dir: str) -> None:
    """Remove SM121A/F from CUDA_PTX_FP4FP6_CVT_ENABLED in float_subbyte.h."""
    target = os.path.join(
        data_dir, "cutlass", "include", "cutlass", "float_subbyte.h"
    )
    if not os.path.exists(target):
        return

    with open(target) as f:
        content = f.read()

    if "SM121 removed" in content:
        return  # already patched

    patched = False
    for suffix in ("A", "F"):
        old = f"defined(CUTLASS_ARCH_MMA_SM121{suffix}_ENABLED)"
        if old in content:
            content = content.replace(f" || \\\n     {old}", "")
            content = content.replace(f"{old} || \\\n     ", "")
            content = content.replace(f" || {old}", "")
            patched = True

    if patched:
        content = content.replace(
            "#  define CUDA_PTX_FP4FP6_CVT_ENABLED 1",
            "/* SM121 removed: no cvt.rn.satfinite.e2m1x2.f32 on GB10 */\n"
            "#  define CUDA_PTX_FP4FP6_CVT_ENABLED 1",
            2,
        )
        with open(target, "w") as f:
            f.write(content)
        logger.debug("GB10 compat: patched %s", target)


def _patch_quantization_utils(data_dir: str) -> None:
    """Exclude SM121 from PTX E2M1 path in TRT-LLM quantization_utils.cuh."""
    target = os.path.join(
        data_dir,
        "csrc",
        "nv_internal",
        "tensorrt_llm",
        "kernels",
        "quantization_utils.cuh",
    )
    if not os.path.exists(target):
        return

    with open(target) as f:
        content = f.read()

    if "__CUDA_ARCH__ != 1210" in content:
        return  # already patched

    old_guard = "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)"
    new_guard = (
        "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)"
        " && (__CUDA_ARCH__ != 1210)"
    )
    if old_guard in content:
        content = content.replace(old_guard, new_guard)
        with open(target, "w") as f:
            f.write(content)
        logger.debug("GB10 compat: patched %s", target)


def _patch_arch_condition(data_dir: str) -> None:
    """Allow SM121 in FlashInfer's arch_condition.h."""
    target = os.path.join(
        data_dir, "include", "flashinfer", "arch_condition.h"
    )
    if not os.path.exists(target):
        return

    with open(target) as f:
        content = f.read()

    if "GB10" in content or "__CUDA_ARCH__ == 1210" in content:
        return  # already patched

    error_line = '#error "Compiling for SM90 or newer'
    if error_line not in content:
        return

    bypass = (
        "// GB10 (SM121): sm_121a is valid, bypass arch-family check\n"
        "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210\n"
        "  // SM121 detected - allow\n"
        "#else\n"
    )
    idx = content.find(error_line)
    block_start = content.rfind("#if", 0, idx)
    if block_start < 0:
        return
    endif_idx = content.find("#endif", idx)
    if endif_idx < 0:
        return
    endif_end = content.find("\n", endif_idx) + 1
    original_block = content[block_start:endif_end]
    patched_block = bypass + original_block + "#endif  // GB10\n"
    content = content[:block_start] + patched_block + content[endif_end:]
    with open(target, "w") as f:
        f.write(content)
    logger.debug("GB10 compat: patched %s", target)


def _copy_fp4_header(data_dir: str) -> None:
    """Copy nv_fp4_dummy.h to FlashInfer include dir for JIT compilation."""
    # Look for the header relative to this file's package root
    _here = os.path.dirname(os.path.abspath(__file__))
    vllm_root = os.path.dirname(_here)  # vllm/
    src = os.path.join(os.path.dirname(vllm_root), "csrc", "nv_fp4_dummy.h")

    dst_dir = os.path.join(data_dir, "include", "flashinfer")
    if not os.path.isdir(dst_dir) or not os.path.exists(src):
        return

    dst = os.path.join(dst_dir, "nv_fp4_dummy.h")
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
        logger.debug("GB10 compat: copied %s -> %s", src, dst)


def _clear_moe_jit_cache() -> None:
    """Clear FlashInfer MoE JIT cache to force recompilation."""
    cache_dir = os.path.expanduser("~/.cache/flashinfer")
    if not os.path.exists(cache_dir):
        return
    for root, dirs, _ in os.walk(cache_dir):
        for d in dirs:
            if "fused_moe" in d or "moe" in d:
                path = os.path.join(root, d)
                shutil.rmtree(path, ignore_errors=True)
                logger.debug("GB10 compat: cleared JIT cache %s", path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@cache
def ensure_flashinfer_sm121_compat() -> None:
    """Apply all SM121 FlashInfer header patches (idempotent, runs once).

    Called automatically by ``vllm.utils.flashinfer.has_flashinfer()`` when
    an SM121 device is detected.  Safe to call multiple times; results are
    cached after the first call.
    """
    data_dir = _find_flashinfer_data_dir()
    if data_dir is None:
        logger.debug(
            "GB10 compat: FlashInfer data dir not found, skipping patches"
        )
        return

    logger.info(
        "GB10 (SM121) detected — applying FlashInfer compatibility patches"
    )
    _patch_float_subbyte(data_dir)
    _patch_quantization_utils(data_dir)
    _patch_arch_condition(data_dir)
    _copy_fp4_header(data_dir)
    _clear_moe_jit_cache()
    logger.info("GB10 compat: FlashInfer patches complete")
