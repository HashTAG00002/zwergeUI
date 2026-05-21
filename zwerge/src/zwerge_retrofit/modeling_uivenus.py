"""
ZwerGe-UI Retrofit: UIVenusRetrofitModel
==========================================
UI-Venus-1.5（Qwen3-VL 系列）retrofit 版本。
架构与 GUI-Owl-1.5 完全相同（同为 36层/hidden=4096/deepstack）。

复用 GUIOwlRetrofitModel 的工厂模式，只改类名。
prompt 差异（无 system message，用 PROMPT_WITH_REFUSAL 包装 user 文本）
完全由 constants.py 中的 MODEL_TYPE_CONSTANTS 控制，model 类无需感知。
"""

from .modeling_guiowl import GUIOwlRetrofitModel, _get_qwen3vl_class
from .modeling_base import RetrofitModelMixin, BaseRetrofitOutput


class UIVenusRetrofitModel(GUIOwlRetrofitModel):
    """
    UI-Venus-1.5（Qwen3-VL）retrofitted with layer-wise coordinate-free grounding head.

    架构与 GUIOwlRetrofitModel 完全相同（共享同一个 _concrete_class 工厂），
    仅在 prompt 层面有差异（由 constants.MODEL_TYPE_CONSTANTS["uivenus"] 控制）。

    from_pretrained / __new__ 均通过 GUIOwlRetrofitModel._get_concrete_class() 路由，
    运行时实例的 __class__.__name__ 会被设为 "UIVenusRetrofitModel" 以便区分。
    """

    # 独立的 concrete class cache（命名不同于 GUI-Owl）
    _concrete_class = None

    @classmethod
    def _get_concrete_class(cls):
        if cls._concrete_class is None:
            # 获取 GUI-Owl 的实现类作为父类
            GuiOwlImpl = GUIOwlRetrofitModel._get_concrete_class()

            class _UIVenusImpl(GuiOwlImpl):
                """UIVenusRetrofitModel 的运行时具体类（与 GUIOwl 共享所有逻辑）。"""
                pass

            _UIVenusImpl.__name__ = "UIVenusRetrofitModel"
            _UIVenusImpl.__qualname__ = "UIVenusRetrofitModel"
            cls._concrete_class = _UIVenusImpl

        return cls._concrete_class
