"""
UI-TARS Retrofit Inference
==========================
Mirrors src/zwerge_retrofit/modeling_uitars.py on the inference side.
"""

from inference_base import RetrofitInference


class UITARSRetrofitInference(RetrofitInference):
    """
    Retrofit inference for UI-TARS-1.5-7B (Qwen2.5-VL).

    patch_size=14: each visual token = 14*2=28 px (Qwen2.5-VL default).
    """
    model_type = "uitars"
    merge_size = 2
    patch_size = 14   # Qwen2.5-VL
