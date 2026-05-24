"""
ZwerGe-UI Retrofit: GUIOwlRetrofitModel
=========================================
GUI-Owl-1.5（Qwen3-VL 系列）retrofit 版本。
继承 RetrofitModelMixin + Qwen3VLForConditionalGeneration。

架构参数（GUI-Owl-1.5-8B-Instruct / UI-Venus-1.5-8B 实测）：
  - num_hidden_layers: 36        （LLM decoder 层数，text_config）
  - hidden_size: 4096
  - patch_size: 16, spatial_merge_size: 2
  - vision_config.depth: 27      （ViT 视觉编码器层数）
  - vision_config.deepstack_visual_indexes: [8, 16, 24]
      ⚠️  这里的 8/16/24 是 ViT（27层）的层号，不是 LLM 层号！
      ViT block 8  → deepstack_visual_embeds[0]  → 注入 LLM decoder 第 0 层之后
      ViT block 16 → deepstack_visual_embeds[1]  → 注入 LLM decoder 第 1 层之后
      ViT block 24 → deepstack_visual_embeds[2]  → 注入 LLM decoder 第 2 层之后
      LLM 第 3~35 层：正常处理，无 deepstack 注入
  - image_token_id: 151655
  - vision_end_token_id: 151653

Qwen3-VL 兼容性要求：
  transformers >= 4.57.1  （qwen3 conda 环境）
  gui_actor 环境（transformers 4.51.3）**不**支持 Qwen3VL，
  本文件使用懒加载（在 class body 以外不 import Qwen3VLForConditionalGeneration），
  确保在 gui_actor 中仍可 import zwerge_retrofit 而不报错。

DeepStack 处理（关键设计）：
  Qwen3-VL 的 deepstack 机制：ViT（27层）在第 8/16/24 层分别输出中间特征
  （vision_config.deepstack_visual_indexes=[8,16,24]），这些特征作为
  deepstack_visual_embeds[0/1/2] 依次注入到 LLM decoder 第 0/1/2 层之后。
  transformers 官方 Qwen3VLForConditionalGeneration.forward() 自动处理 deepstack，
  无需在 retrofit 代码中手动传递 deepstack_visual_embeds。

Action format (GUI-Owl-1.5 retrofit prefill):
  <tool_call>
  {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [
  <|ground|><|pointer_start|><|pointer_pad|><|pointer_end|>]}}
  </tool_call>
"""

from typing import List, Optional, Tuple, Union

import torch

from .modeling_base import (
    RetrofitModelMixin,
    BaseRetrofitOutput,
)


def _get_qwen3vl_class():
    """懒加载 Qwen3VLForConditionalGeneration（仅在需要时 import，避免旧 env 报错）。"""
    try:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    except ImportError:
        raise ImportError(
            "Qwen3VLForConditionalGeneration not found in current transformers version. "
            "GUI-Owl-1.5 / UI-Venus-1.5 require transformers>=4.57.1. "
            "Please use the 'qwen3' conda environment: "
            "conda activate qwen3"
        )


class GUIOwlRetrofitModel(RetrofitModelMixin):
    """
    GUI-Owl-1.5（Qwen3-VL, 36层, hidden=4096）retrofitted with
    layer-wise coordinate-free grounding head.

    使用懒加载动态继承 Qwen3VLForConditionalGeneration，
    确保在旧 transformers 环境中 import 本文件不会报错。

    forward() 直接调用 super().forward(output_hidden_states=True)，
    官方实现处理 DeepStack、visual embedding、RoPE 等所有细节。

    Usage:
      from zwerge_retrofit import get_model_class
      ModelClass = get_model_class("guiowl")  # 触发懒加载
      model = ModelClass.from_pretrained(model_path, config=config)
    """

    # __init_subclass__ / metaclass 要求 Qwen3VL 在类定义时就存在，
    # 所以我们用工厂函数在第一次实例化时动态创建真正的类。
    # GUIOwlRetrofitModel 本身是一个"壳"，真正的实例化通过 __new__ 返回。
    _concrete_class = None   # cache

    def __class_getitem__(cls, item):
        return cls

    @classmethod
    def _get_concrete_class(cls):
        """返回（或创建）真正继承 Qwen3VL 的具体类。"""
        if cls._concrete_class is None:
            Qwen3VL = _get_qwen3vl_class()

            class _GUIOwlImpl(RetrofitModelMixin, Qwen3VL):
                """真正的运行时类：RetrofitModelMixin + Qwen3VLForConditionalGeneration。"""

                def __init__(self, config, *args, **kwargs):
                    super().__init__(config, *args, **kwargs)
                    self._init_retrofit_from_config(config)
                    self.post_init()

                def _init_retrofit_from_config(self, config) -> None:
                    """覆盖以读取 Qwen3-VL text_config 中的 hidden_size / num_layers。"""
                    # Qwen3-VL 将 hidden_size / num_hidden_layers 等字段放在 text_config 子配置里
                    text_cfg = getattr(config, "text_config", config)
                    if not hasattr(config, "hidden_size"):
                        config.hidden_size = getattr(text_cfg, "hidden_size", 4096)
                    if not hasattr(config, "num_hidden_layers"):
                        config.num_hidden_layers = getattr(text_cfg, "num_hidden_layers", 36)
                    super()._init_retrofit_from_config(config)
                    # vision_end_token_id 在 Qwen3-VL 是顶层字段
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
                ) -> Tuple[torch.Tensor, ...]:
                    """Qwen3-VL: 调用官方 super().forward(output_hidden_states=True)。
                    官方实现自动处理 DeepStack、visual embedding、RoPE 等。"""
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

                    # Official Qwen3-VL forward handles DeepStack, visual embedding,
                    # RoPE, etc. output_hidden_states=True gives us all layer hidden states.
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

            # 复制类名使 HuggingFace 保存逻辑正常工作
            _GUIOwlImpl.__name__ = "GUIOwlRetrofitModel"
            _GUIOwlImpl.__qualname__ = "GUIOwlRetrofitModel"
            cls._concrete_class = _GUIOwlImpl

        return cls._concrete_class

    def __new__(cls, *args, **kwargs):
        """
        拦截实例化，返回真正继承 Qwen3VL 的类的实例。
        这样 from_pretrained() 也能正常工作。
        """
        concrete = cls._get_concrete_class()
        instance = object.__new__(concrete)
        return instance

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """代理到具体类的 from_pretrained。"""
        return cls._get_concrete_class().from_pretrained(*args, **kwargs)
