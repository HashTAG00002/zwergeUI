# ZwerGe-UI Retrofit Package
# Coordinate-Free Grounding Retrofit for Native GUI Agents
#
# 兼容性说明：
#   - UITARSRetrofitModel (uitars):     需要 transformers>=4.51.3，Qwen2.5-VL
#   - GUIOwl7BRetrofitModel (guiowl7b): 需要 transformers>=4.51.3，Qwen2.5-VL（控制变量）
#   - GUIOwlRetrofitModel (guiowl):     需要 transformers>=4.57.1，Qwen3-VL (qwen3 conda env)
#   - UIVenusRetrofitModel (uivenus):   同上
#
# __init__.py 仅导出通用基础组件和工厂函数，具体模型类使用懒加载，
# 这样在 gui_actor (transformers 4.51.3) 环境中导入本包不会因 Qwen3VL 不存在而报错。

from .modeling_base import (
    AnchorStrategy,
    BaseRetrofitOutput,
    LayerWiseGroundingHead,
    RetrofitModelMixin,
)


def get_model_class(model_type: str):
    """
    工厂函数：返回指定 model_type 的 retrofit 模型类。
    使用懒加载（在函数内部 import），确保：
      - 在 gui_actor (transformers 4.51.3) 中仍可正常 import zwerge_retrofit 包
      - 只有当用户真正需要 guiowl/uivenus 时才触发 Qwen3VL import
    """
    if model_type == "uitars":
        from .modeling_uitars import UITARSRetrofitModel
        return UITARSRetrofitModel
    elif model_type == "guiowl7b":
        # GUI-Owl-7B: Qwen2.5-VL + GUI-Owl prompt（控制变量，与 guiowl1.5 对比）
        from .modeling_guiowl7b import GUIOwl7BRetrofitModel
        return GUIOwl7BRetrofitModel
    elif model_type == "guiowl":
        from .modeling_guiowl import GUIOwlRetrofitModel
        return GUIOwlRetrofitModel
    elif model_type == "uivenus":
        from .modeling_uivenus import UIVenusRetrofitModel
        return UIVenusRetrofitModel
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: ['uitars', 'guiowl7b', 'guiowl', 'uivenus']"
        )


# 兼容旧代码直接 from zwerge_retrofit import UITARSRetrofitModel 的 import 方式
def __getattr__(name):
    if name == "UITARSRetrofitModel":
        from .modeling_uitars import UITARSRetrofitModel
        return UITARSRetrofitModel
    if name == "RetrofitOutputWithPast":
        from .modeling_uitars import RetrofitOutputWithPast
        return RetrofitOutputWithPast
    if name == "GUIOwl7BRetrofitModel":
        from .modeling_guiowl7b import GUIOwl7BRetrofitModel
        return GUIOwl7BRetrofitModel
    if name == "GUIOwlRetrofitModel":
        from .modeling_guiowl import GUIOwlRetrofitModel
        return GUIOwlRetrofitModel
    if name == "UIVenusRetrofitModel":
        from .modeling_uivenus import UIVenusRetrofitModel
        return UIVenusRetrofitModel
    raise AttributeError(f"module 'zwerge_retrofit' has no attribute '{name}'")


MODEL_REGISTRY = {
    "uitars":    "UITARSRetrofitModel",
    "guiowl7b":  "GUIOwl7BRetrofitModel",   # Qwen2.5-VL + GUI-Owl prompt（控制变量）
    "guiowl":    "GUIOwlRetrofitModel",
    "uivenus":   "UIVenusRetrofitModel",
}
