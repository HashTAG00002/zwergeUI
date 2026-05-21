"""
ZwerGe-UI Retrofit: GUIOwlRetrofitModel
=========================================
GUI-Owl-1.5（Qwen3-VL 系列）retrofit 版本。
继承 RetrofitModelMixin + Qwen3VLForConditionalGeneration。

架构参数（GUI-Owl-1.5-8B-Instruct / UI-Venus-1.5-8B 实测）：
  - num_hidden_layers: 36
  - hidden_size: 4096
  - patch_size: 16, spatial_merge_size: 2
  - deepstack_visual_indexes: [8, 16, 24]  ← 视觉特征在中间层重注入
  - image_token_id: 151655
  - vision_end_token_id: 151653

Qwen3-VL 兼容性要求：
  transformers >= 4.57.1  （qwen3 / qwen3-verl conda 环境）
  gui_actor 环境（transformers 4.51.3）**不**支持 Qwen3VL，
  本文件使用懒加载（在 class body 以外不 import Qwen3VLForConditionalGeneration），
  确保在 gui_actor 中仍可 import zwerge_retrofit 而不报错。

deepstack 处理（关键设计）：
  Qwen3-VL 的 deepstack 在 transformer 层 8/16/24 重注入视觉特征。
  如果直接调用 model.model(inputs_embeds=...) 绕过 Qwen3VLForConditionalGeneration.forward()，
  deepstack 注入不会发生，导致 hidden states 不正确。
  因此 _forward_hidden_states_for_grounding() 必须通过 Qwen3VLForConditionalGeneration.forward()
  来正确触发 deepstack。

Action format (GUI-Owl-1.5 retrofit prefill):
  <tool_call>
  {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [
  <|ground|><|pointer_start|><|pointer_pad|><|pointer_end|>]}}
  </tool_call>
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

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

    关键特性：
      - _forward_hidden_states_for_grounding() 通过 Qwen3VL.forward() 正确触发 deepstack
      - forward() 调用 super().forward() 由 Qwen3VL 处理所有 backbone 逻辑

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
                    # ── Patch Conv3d → Linear in VisionPatchEmbed ─────────────────────
                    # Called AFTER post_init / weight loading.
                    # Qwen3VLVisionPatchEmbed uses Conv3d(in_ch, out_ch, kernel=tps×ps×ps,
                    # stride=tps×ps×ps) which is mathematically equivalent to
                    # Linear(in_ch*tps*ps*ps, out_ch) when applied to flat [N, flat_dim] inputs
                    # from the image processor.
                    #
                    # BUT Conv3d with batch_size=N_patches (11360) and spatial=1×1×1 is
                    # >100 000× SLOWER than the equivalent Linear on GPU:
                    #   Conv3d:  ~93 000 ms   (not optimized by cuDNN for this degenerate case)
                    #   Linear:  <1 ms        (GEMM on [N, in_flat] × [in_flat, out_ch])
                    #
                    # Fix: monkey-patch PatchEmbed.forward to use F.linear instead of Conv3d.
                    # The Conv3d weights are kept in place (shape unchanged) so checkpoint
                    # loading / saving is unaffected; we only change how forward() uses them.
                    self._patch_embed_conv3d_to_linear_forward()

                def _patch_embed_conv3d_to_linear_forward(self) -> None:
                    """
                    Monkey-patch Qwen3VLVisionPatchEmbed.forward to use F.linear (gemm)
                    instead of Conv3d.  The Conv3d weight tensor is KEPT (shape unchanged)
                    so checkpoint loading/saving continues to work normally.

                    Mathematical equivalence:
                      Conv3d(weight=[O, C, T, H, W], bias=[O])(input=[N, C, T, H, W])
                        == F.linear(input.view(N, C*T*H*W), weight.view(O, C*T*H*W), bias)
                    when stride == kernel_size (i.e., each kernel fires exactly once per patch).
                    """
                    import types, torch.nn.functional as F_nn
                    try:
                        patch_embed = self.model.visual.patch_embed
                        proj = patch_embed.proj
                        if not isinstance(proj, nn.Conv3d):
                            return  # already patched or unexpected type – skip silently

                        def _fast_patch_embed_forward(
                            self_pe,
                            hidden_states: torch.Tensor,
                        ) -> torch.Tensor:
                            # hidden_states: [N_patches, C*T*H*W]  (flat, from processor)
                            # proj.weight:   [out_ch, C, T, H, W]
                            # proj.bias:     [out_ch] or None
                            target_dtype = self_pe.proj.weight.dtype
                            x = hidden_states.to(dtype=target_dtype)   # [N, flat]
                            w = self_pe.proj.weight.view(self_pe.proj.weight.shape[0], -1)  # [out_ch, flat]
                            b = self_pe.proj.bias  # [out_ch] or None
                            return F_nn.linear(x, w, b)                # [N, out_ch]

                        patch_embed.forward = types.MethodType(
                            _fast_patch_embed_forward, patch_embed
                        )
                    except Exception as exc:
                        import warnings
                        warnings.warn(
                            f"[GUIOwl] _patch_embed_conv3d_to_linear_forward failed: {exc}. "
                            "Vision encoding will still work but will be ~90 000× slower. "
                            "Check that model.visual.patch_embed.proj is nn.Conv3d."
                        )

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

                def _run_language_model(self, input_ids, attention_mask, pixel_values, image_grid_thw):
                    """
                    直接调用 Qwen3-VL 内层 language_model，使用 forward hook 抓取 probe 层
                    的 hidden state，完全避免 output_hidden_states=True。

                    为什么不用 output_hidden_states=True：
                      - output_hidden_states=True 触发 @check_model_inputs 的 37 层额外处理
                      - 与 gradient_checkpointing + FA2 叠加时，每 step 需存/重算全部 37 个
                        hidden state tensor（O(37) 额外内存 + backward recompute 开销）
                      - backbone 完全冻结，根本不需要这些 hidden state 上的梯度流

                    hook 方案：
                      - 只在 probe 层注册 forward hook（通常 15 层）
                      - hook 捕获 hidden state 后立即 .detach()（backbone 冻结，无需反传）
                      - language model 以 output_hidden_states=False 运行，无任何额外开销
                      - backward 只经过 grounding head（hook 的 detach 已切断 backbone 梯度链路）

                    Returns: (lm_out, all_hidden_states)
                      lm_out:            language_model 的 ModelOutput（last_hidden_state 等）
                      all_hidden_states: 稀疏 tuple，only probe 层位置非 None，
                                         供 LayerWiseGroundingHead 直接使用
                                         （all_hidden_states[layer_idx+1] = hs for each probe layer）
                    """
                    # 1. Embed input tokens
                    inputs_embeds = self.model.language_model.embed_tokens(input_ids)

                    # 2. Process visual features (deepstack-aware)
                    visual_pos_masks = None
                    deepstack_visual_embeds = None
                    if pixel_values is not None:
                        image_embeds, deepstack_image_embeds = self.model.get_image_features(
                            pixel_values, image_grid_thw
                        )
                        image_embeds_cat = torch.cat(image_embeds, dim=0).to(
                            inputs_embeds.device, inputs_embeds.dtype
                        )
                        image_mask, _ = self.model.get_placeholder_mask(
                            input_ids,
                            inputs_embeds=inputs_embeds,
                            image_features=image_embeds_cat,
                        )
                        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat)
                        visual_pos_masks = image_mask[..., 0].bool()
                        deepstack_visual_embeds = deepstack_image_embeds

                    # 3. Compute RoPE position ids
                    position_ids, _ = self.model.get_rope_index(
                        input_ids, image_grid_thw, None, attention_mask
                    )

                    # 4. Register hooks on probe layers; run LM with output_hidden_states=False
                    probe_layers = list(self.layerwise_grounding_head.probe_layers)
                    hidden_states_cache = {}
                    hooks = []
                    for layer_idx in probe_layers:
                        def _make_hook(idx):
                            def _hook(module, inputs, output):
                                hs = output[0] if isinstance(output, tuple) else output
                                if not torch.is_tensor(hs) or hs.ndim != 3:
                                    raise RuntimeError(
                                        f"Probe layer {idx} hook: expected 3-D tensor "
                                        f"[B, T, D], got {type(hs).__name__} "
                                        f"shape={getattr(hs, 'shape', 'N/A')}. "
                                        "Check that probe_layers indices are decoder layers."
                                    )
                                # .detach(): backbone frozen; no grads through backbone.
                                # Grounding head parameters get gradients from their own
                                # forward computation starting from these detached inputs.
                                hidden_states_cache[idx] = hs.detach()
                            return _hook
                        hooks.append(
                            self.model.language_model.layers[layer_idx].register_forward_hook(
                                _make_hook(layer_idx)
                            )
                        )

                    try:
                        lm_out = self.model.language_model(
                            input_ids=None,
                            inputs_embeds=inputs_embeds,
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                            past_key_values=None,
                            use_cache=False,
                            output_attentions=False,
                            output_hidden_states=False,   # no 37-tensor overhead
                            return_dict=True,
                            visual_pos_masks=visual_pos_masks,
                            deepstack_visual_embeds=deepstack_visual_embeds,
                        )
                    finally:
                        for h in hooks:
                            h.remove()

                    # 5. Build sparse all_hidden_states tuple for LayerWiseGroundingHead.
                    # LayerWiseGroundingHead.forward uses: hs = all_hidden_states[layer_idx + 1]
                    assert len(hidden_states_cache) == len(probe_layers), (
                        f"Expected {len(probe_layers)} hook captures, got {len(hidden_states_cache)}. "
                        "Check that probe_layers indices are valid for this model."
                    )
                    max_idx = max(probe_layers) + 2   # +1 for layer→hs offset, +1 for length
                    hs_list = [None] * max_idx
                    for layer_idx, hs in hidden_states_cache.items():
                        hs_list[layer_idx + 1] = hs
                    all_hidden_states = tuple(hs_list)

                    return lm_out, all_hidden_states

                # ── grounding inference path (eval/vis) ───────────────────────────────
                @torch.no_grad()
                def _forward_hidden_states_for_grounding(
                    self,
                    input_ids: torch.LongTensor,
                    attention_mask,
                    pixel_values,
                    image_grid_thw,
                    device,
                ) -> Tuple[torch.Tensor, ...]:
                    """Qwen3-VL deepstack: 直接调 language_model，hook 抓 probe 层 hidden states。"""
                    _, all_hidden_states = self._run_language_model(
                        input_ids, attention_mask, pixel_values, image_grid_thw,
                    )
                    return all_hidden_states

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

                    # hook 方案：避免 output_hidden_states=True 的存储/recompute 开销
                    lm_out, all_hidden_states = self._run_language_model(
                        input_ids, attention_mask, pixel_values, image_grid_thw,
                    )
                    logits = self.lm_head(lm_out.last_hidden_state)

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
                            past_key_values=None,
                            hidden_states=None,
                            attentions=None,
                            rope_deltas=None,
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
