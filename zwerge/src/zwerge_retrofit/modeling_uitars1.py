"""
ZwerGe-UI Retrofit: UITARS1RetrofitModel
==========================================
UI-TARS-7B-SFT (Qwen2VLForConditionalGeneration) retrofit 版本。

架构参数（UI-TARS-7B-SFT 实测）：
  - model_type: qwen2_vl
  - architectures: Qwen2VLForConditionalGeneration
  - num_hidden_layers: 28
  - hidden_size: 3584
  - patch_size: 14, spatial_merge_size: 2
  - Probe 层：layers 14-27（last 14 of 28，last 1/2）

与 UITARSRetrofitModel（Qwen2_5_VLForConditionalGeneration）的关系：
  Qwen2VLForConditionalGeneration 与 Qwen2_5_VLForConditionalGeneration **不**存在继承关系，
  是完全独立的两个类（分别在 transformers/models/qwen2_vl/ 和 transformers/models/qwen2_5_vl/）。
  API 接口完全兼容（forward 参数相同），但必须使用各自的类。
  因此 UITARS1RetrofitModel 有独立的 lazy-loader（_get_qwen2vl_class），
  不复用 UITARSRetrofitModel。

Prompt 格式：UI-TARS-1.5 格式（与 uitars 完全相同）：
  click(start_box='<|ground|><|pointer_start|><|pointer_pad|><|pointer_end|>')
  prompt 差异完全由 constants.py 中的 MODEL_TYPE_CONSTANTS["uitars1"] 控制。

坐标系：relative 1000（与 guiowl/uivenus/qwen35 相同，与 uitars 的 absolute pixel 不同）。

兼容性要求：
  transformers 支持 Qwen2VLForConditionalGeneration 的最低版本（qwen2 conda 环境）。
  本文件使用懒加载（在 class body 以外不 import Qwen2VLForConditionalGeneration），
  确保在其他环境中 import zwerge_retrofit 不报错。

Usage:
  from zwerge_retrofit import get_model_class
  ModelClass = get_model_class("uitars1")   # 触发懒加载
  model = ModelClass.from_pretrained(model_path, config=config)
"""

from typing import Tuple

import torch

from .modeling_base import RetrofitModelMixin, BaseRetrofitOutput


def _get_qwen2vl_class():
    """懒加载 Qwen2VLForConditionalGeneration（仅在需要时 import，避免旧 env 报错）。"""
    try:
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration
    except ImportError:
        raise ImportError(
            "Qwen2VLForConditionalGeneration not found in current transformers version. "
            "UI-TARS-7B requires a transformers version that supports Qwen2-VL. "
            "Please use the 'qwen2' conda environment: "
            "conda activate qwen2"
        )


class UITARS1RetrofitModel(RetrofitModelMixin):
    """
    UI-TARS-7B-SFT (Qwen2-VL) retrofitted with layer-wise coordinate-free grounding head.

    使用懒加载动态继承 Qwen2VLForConditionalGeneration，
    确保在旧 transformers 环境中 import 本文件不会报错。

    Prompt 格式与 uitars (UI-TARS-1.5) 完全相同，但坐标系为 relative 1000（非 absolute pixel）。

    forward() 直接调用 super().forward(output_hidden_states=True)，
    官方实现处理 visual embedding、RoPE 等所有细节。

    Usage:
      from zwerge_retrofit import get_model_class
      ModelClass = get_model_class("uitars1")  # 触发懒加载
      model = ModelClass.from_pretrained(model_path, config=config)
    """

    _concrete_class = None

    def __class_getitem__(cls, item):
        return cls

    @classmethod
    def _get_concrete_class(cls):
        """返回（或创建）真正继承 Qwen2VLForConditionalGeneration 的具体类。"""
        if cls._concrete_class is None:
            Qwen2VL = _get_qwen2vl_class()

            class _UITARS1Impl(RetrofitModelMixin, Qwen2VL):
                """真正的运行时类：RetrofitModelMixin + Qwen2VLForConditionalGeneration。"""

                def __init__(self, config, *args, **kwargs):
                    super().__init__(config, *args, **kwargs)
                    self._init_retrofit_from_config(config)
                    self.post_init()

                def _init_retrofit_from_config(self, config) -> None:
                    """读取 Qwen2-VL config 中的 hidden_size / num_layers。"""
                    # Qwen2-VL 将 hidden_size / num_hidden_layers 放在顶层 config
                    if not hasattr(config, "hidden_size"):
                        config.hidden_size = 3584
                    if not hasattr(config, "num_hidden_layers"):
                        config.num_hidden_layers = 28
                    super()._init_retrofit_from_config(config)
                    # vision_end_token_id 在 Qwen2-VL 是顶层字段
                    if self._vision_end_token_id is None:
                        self._vision_end_token_id = getattr(config, "vision_end_token_id", None)

                # ── Grounding inference path (eval/vis) ──────────────────────────────
                @torch.no_grad()
                def _forward_hidden_states_for_grounding(
                    self,
                    input_ids: torch.LongTensor,
                    attention_mask,
                    pixel_values,
                    image_grid_thw,
                    device,
                    mm_token_type_ids=None,
                ) -> Tuple[torch.Tensor, ...]:
                    """Qwen2-VL: 调用官方 super().forward(output_hidden_states=True)。
                    mm_token_type_ids: Qwen3.5 专属字段，Qwen2-VL 不需要，忽略即可。
                    """
                    outputs = super().forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        output_hidden_states=True,
                        output_attentions=False,
                        return_dict=True,
                        use_cache=False,
                    )
                    return outputs.hidden_states

                # ── Training forward ──────────────────────────────────────────────────
                def forward(
                    self,
                    input_ids=None,
                    attention_mask=None,
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds=None,
                    labels=None,
                    use_cache=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None,
                    pixel_values=None,
                    image_grid_thw=None,
                    rope_deltas=None,
                    cache_position=None,
                    ground_token_indices=None,
                    multi_patch_labels=None,
                    verbose=False,
                    **extra_kwargs,
                ):
                    return_dict = (
                        return_dict if return_dict is not None
                        else self.config.use_return_dict
                    )

                    outputs = super().forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        labels=None,
                        use_cache=use_cache,
                        output_attentions=False,
                        output_hidden_states=True,
                        return_dict=True,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        rope_deltas=rope_deltas,
                        cache_position=cache_position,
                        **extra_kwargs,
                    )

                    all_hidden_states = outputs.hidden_states
                    logits = outputs.logits

                    lm_loss = None
                    if labels is not None and self.lm_loss_weight > 0:
                        from torch.nn import CrossEntropyLoss
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        lm_loss = CrossEntropyLoss()(
                            shift_logits.view(-1, shift_logits.shape[-1]),
                            shift_labels.view(-1).to(shift_logits.device),
                        )

                    grounding_loss, all_scores, all_weights, all_anchors = (
                        self._compute_grounding_loss(
                            all_hidden_states=all_hidden_states,
                            input_ids=input_ids,
                            logits=logits,
                            ground_token_indices=ground_token_indices,
                            multi_patch_labels=multi_patch_labels,
                            verbose=verbose,
                        )
                    )

                    total_loss = None
                    if lm_loss is not None and grounding_loss is not None:
                        total_loss = (
                            self.lm_loss_weight * lm_loss
                            + self.grounding_loss_weight * grounding_loss
                        )
                    elif grounding_loss is not None:
                        total_loss = self.grounding_loss_weight * grounding_loss
                    elif lm_loss is not None:
                        total_loss = lm_loss

                    if return_dict:
                        return BaseRetrofitOutput(
                            lm_loss=lm_loss,
                            grounding_loss=grounding_loss,
                            grounding_scores=all_scores,
                            layer_weights=all_weights,
                            anchor_positions=(
                                all_anchors if multi_patch_labels is not None else []
                            ),
                            loss=total_loss,
                            logits=logits,
                            past_key_values=outputs.past_key_values,
                            hidden_states=None,
                            attentions=None,
                            rope_deltas=(
                                outputs.rope_deltas
                                if hasattr(outputs, "rope_deltas") else None
                            ),
                        )
                    else:
                        return (total_loss, logits) if total_loss is not None else (logits,)

            _UITARS1Impl.__name__ = "UITARS1RetrofitModel"
            _UITARS1Impl.__qualname__ = "UITARS1RetrofitModel"
            cls._concrete_class = _UITARS1Impl

        return cls._concrete_class

    def __new__(cls, *args, **kwargs):
        """拦截实例化，返回真正继承 Qwen2VLForConditionalGeneration 的类的实例。"""
        concrete = cls._get_concrete_class()
        instance = object.__new__(concrete)
        return instance

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """代理到具体类的 from_pretrained。"""
        return cls._get_concrete_class().from_pretrained(*args, **kwargs)
