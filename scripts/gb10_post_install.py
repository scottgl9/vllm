#!/usr/bin/env python3
"""
GB10 (SM121) post-install script for FlashInfer compatibility.

SM121 has FP4 tensor core MMA but lacks the cvt.rn.satfinite.e2m1x2.f32
PTX instruction for float-to-E2M1 conversion. FlashInfer's JIT-compiled
kernels reference this instruction via CUTLASS headers, causing build
failures at runtime.

This script patches installed FlashInfer headers to:
1. Remove SM121A from CUDA_PTX_FP4FP6_CVT_ENABLED in float_subbyte.h
2. Add software E2M1 fallbacks in TRT-LLM quantization_utils.cuh
3. Patch FlashInfer arch check to allow SM121
4. Clear FlashInfer JIT cache

Run after: pip install flashinfer-python && pip install -e .
Usage: python scripts/gb10_post_install.py
"""

import os
import shutil
import site
import sys


def find_flashinfer_data_dir():
    """Find the FlashInfer data directory in site-packages."""
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        candidate = os.path.join(sp, "flashinfer", "data")
        if os.path.isdir(candidate):
            return candidate
    # Also check the current venv
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        for pyver in ["python3.12", "python3.11", "python3.13", "python3.10"]:
            candidate = os.path.join(venv, "lib", pyver, "site-packages",
                                     "flashinfer", "data")
            if os.path.isdir(candidate):
                return candidate
    return None


def patch_float_subbyte(data_dir):
    """Remove SM121 from CUDA_PTX_FP4FP6_CVT_ENABLED."""
    target = os.path.join(data_dir, "cutlass", "include", "cutlass",
                          "float_subbyte.h")
    if not os.path.exists(target):
        print(f"  SKIP: {target} not found")
        return

    with open(target) as f:
        content = f.read()

    if "SM121 removed" in content:
        print("  float_subbyte.h: already patched")
        return

    patched = False
    # Remove SM121A from the CUDA_PTX_FP4FP6_CVT_ENABLED macro
    for suffix in ["A", "F"]:
        old = f"defined(CUTLASS_ARCH_MMA_SM121{suffix}_ENABLED)"
        if old in content:
            # Remove the pattern and clean up surrounding ||
            content = content.replace(f" || \\\n     {old}", "")
            content = content.replace(f"{old} || \\\n     ", "")
            content = content.replace(f" || {old}", "")
            patched = True

    if patched:
        # Add a comment noting the patch
        content = content.replace(
            "#  define CUDA_PTX_FP4FP6_CVT_ENABLED 1",
            "/* SM121 removed: no cvt.rn.satfinite.e2m1x2.f32 on GB10 */\n"
            "#  define CUDA_PTX_FP4FP6_CVT_ENABLED 1",
            2  # patch both A and F blocks
        )
        with open(target, "w") as f:
            f.write(content)
        print(f"  Patched: {target}")
    else:
        print("  float_subbyte.h: SM121 pattern not found (may be OK)")


def patch_quantization_utils(data_dir):
    """Exclude SM121 from PTX E2M1 path in TRT-LLM quantization_utils.cuh."""
    target = os.path.join(data_dir, "csrc", "nv_internal", "tensorrt_llm",
                          "kernels", "quantization_utils.cuh")
    if not os.path.exists(target):
        print(f"  SKIP: {target} not found")
        return

    with open(target) as f:
        content = f.read()

    old_guard = "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)"
    new_guard = ("#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)"
                 " && (__CUDA_ARCH__ != 1210)")

    if "__CUDA_ARCH__ != 1210" in content:
        print("  quantization_utils.cuh: already patched")
        return

    if old_guard in content:
        content = content.replace(old_guard, new_guard)
        with open(target, "w") as f:
            f.write(content)
        print(f"  Patched: {target}")
    else:
        print("  quantization_utils.cuh: guard pattern not found")


def patch_arch_condition(data_dir):
    """Allow SM121 in FlashInfer's architecture check."""
    target = os.path.join(data_dir, "include", "flashinfer",
                          "arch_condition.h")
    if not os.path.exists(target):
        print(f"  SKIP: {target} not found")
        return

    with open(target) as f:
        content = f.read()

    if "GB10" in content or "__CUDA_ARCH__ == 1210" in content:
        print("  arch_condition.h: already patched")
        return

    # Add SM121 bypass before the #error directive
    error_line = '#error "Compiling for SM90 or newer'
    if error_line in content:
        bypass = (
            "// GB10 (SM121): sm_121a is valid, bypass arch-family check\n"
            "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210\n"
            "  // SM121 detected - allow\n"
            "#else\n"
        )
        # Find the #if block containing the #error and wrap it
        idx = content.find(error_line)
        # Find the start of the #if block
        block_start = content.rfind("#if", 0, idx)
        if block_start >= 0:
            # Find the end of the #endif
            endif_idx = content.find("#endif", idx)
            if endif_idx >= 0:
                endif_end = content.find("\n", endif_idx) + 1
                original_block = content[block_start:endif_end]
                patched_block = bypass + original_block + "#endif  // GB10\n"
                content = content[:block_start] + patched_block + content[endif_end:]

                with open(target, "w") as f:
                    f.write(content)
                print(f"  Patched: {target}")
                return

    print("  arch_condition.h: #error pattern not found (may be OK)")


def copy_fp4_header(data_dir):
    """Copy nv_fp4_dummy.h to FlashInfer include dir for JIT compilation."""
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       "csrc", "nv_fp4_dummy.h")
    if not os.path.exists(src):
        # Try relative to script location
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "csrc", "nv_fp4_dummy.h")

    dst_dir = os.path.join(data_dir, "include", "flashinfer")
    if not os.path.isdir(dst_dir):
        print(f"  SKIP: {dst_dir} not found")
        return

    dst = os.path.join(dst_dir, "nv_fp4_dummy.h")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied: {src} -> {dst}")
    else:
        print(f"  SKIP: {src} not found")


def clear_jit_cache():
    """Clear FlashInfer JIT cache to force recompilation."""
    cache_dir = os.path.expanduser("~/.cache/flashinfer")
    if os.path.exists(cache_dir):
        # Only remove fused_moe caches, not everything
        cleared = False
        for root, dirs, files in os.walk(cache_dir):
            for d in dirs:
                if "fused_moe" in d or "moe" in d:
                    path = os.path.join(root, d)
                    shutil.rmtree(path)
                    print(f"  Cleared: {path}")
                    cleared = True
        if not cleared:
            print("  No MoE JIT cache found")
    else:
        print("  No JIT cache found (first run)")


def main():
    print("=== GB10 (SM121) Post-Install FlashInfer Patches ===")
    print()

    data_dir = find_flashinfer_data_dir()
    if data_dir is None:
        print("ERROR: FlashInfer data directory not found.")
        print("Install FlashInfer first: pip install flashinfer-python --pre")
        sys.exit(1)
    print(f"FlashInfer data dir: {data_dir}")
    print()

    print("[1/5] Patching CUTLASS float_subbyte.h...")
    patch_float_subbyte(data_dir)
    print()

    print("[2/5] Patching TRT-LLM quantization_utils.cuh...")
    patch_quantization_utils(data_dir)
    print()

    print("[3/5] Patching FlashInfer arch_condition.h...")
    patch_arch_condition(data_dir)
    print()

    print("[4/5] Copying nv_fp4_dummy.h to FlashInfer includes...")
    copy_fp4_header(data_dir)
    print()

    print("[5/5] Clearing FlashInfer JIT cache...")
    clear_jit_cache()
    print()

    print("Done. FlashInfer will use software E2M1 on SM121.")


if __name__ == "__main__":
    main()
