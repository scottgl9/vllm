# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.vllm_flash_attn.flash_attn_interface import (
    FA2_AVAILABLE,
    FA3_AVAILABLE,
    fa_version_unsupported_reason,
    flash_attn_varlen_func,
    get_scheduler_metadata,
    is_fa_version_supported,
)

if not (FA2_AVAILABLE or FA3_AVAILABLE):
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "vllm.vllm_flash_attn: neither _vllm_fa2_C nor _vllm_fa3_C is "
        "available — FlashInfer will be used as attention backend. "
        "Rebuild vllm with CUDA torch active to enable FA extensions."
    )

__all__ = [
    "fa_version_unsupported_reason",
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "is_fa_version_supported",
]
