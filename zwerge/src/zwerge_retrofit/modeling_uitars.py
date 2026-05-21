"""
ZwerGe-UI Retrofit: UITARSRetrofitModel
=========================================
UI-TARS-1.5-7B（Qwen2.5-VL 系列）retrofit 版本。
继承 RetrofitModelMixin（modeling_base.py）+ Qwen2_5_VLForConditionalGeneration。

所有通用 grounding 组件（LayerWiseGroundingHead、AnchorStrategy 等）均在
modeling_base.py 中定义，此文件只包含 Qwen2.5-VL 特有的部分：
  1. UITARSRetrofitModel   — 模型主体，forward() 处理 Qwen2.5-VL 特有参数
  2. RetrofitOutputWithPast — 向后兼容别名（= BaseRetrofitOutput）

Action format (UI-TARS-1.5 真实固化格式，issue #183/#138 确认):
  click(start_box='<|ground|><|pointer_start|><|pointer_pad|><|pointer_end|>')
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

from .modeling_base import (
    RetrofitModelMixin,
    BaseRetrofitOutput,
)

# Backward-compat alias so existing code that imports RetrofitOutputWithPast still works
RetrofitOutputWithPast = BaseRetrofitOutput


# =============================================================================
# UITARSRetrofitModel
# =============================================================================

class UITARSRetrofitModel(RetrofitModelMixin, Qwen2_5_VLForConditionalGeneration):
    """
    UI-TARS-1.5-7B (or any Qwen2.5-VL-based GUI agent) retrofitted with
    a lightweight layer-wise coordinate-free grounding head.

    Key design decisions:
      1. BACKBONE FROZEN: no gradient through backbone params
      2. QUERY ANCHOR: <|ground|> at pre-coordinate action-prefix position
      3. FA2 COMPATIBLE: output_hidden_states=True, output_attentions=False
      4. LAYER-WISE PROBING: per-layer probe reads hidden states at probe_layers
      5. LEARNED FUSION: readiness scorer learns which layer is most "ready"

    Usage:
      model = UITARSRetrofitModel.from_pretrained(model_path, config=config)
      model.setup_special_token_ids(ground_token_id, pointer_start_token_id)
      model.reset_loss_weights(grounding_loss_weight=1.0, lm_loss_weight=0.0)
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._init_retrofit_from_config(config)
        self.post_init()

    # ─────────────────────────────────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        # ── Retrofit grounding supervision ──────────────────────────────────
        ground_token_indices: Optional[List[Optional[int]]] = None,
        multi_patch_labels: Optional[List[Optional[torch.Tensor]]] = None,
        # Legacy compat (unused but kept for API stability)
        visual_token_indices_of_coordinates: Optional[List[torch.Tensor]] = None,
        coordinates: Optional[List[Tuple[float, float]]] = None,
        verbose: bool = False,
    ) -> Union[Tuple, BaseRetrofitOutput]:
        """
        Full forward pass.

        When multi_patch_labels is provided: computes layerwise grounding loss.
        When labels is provided and lm_loss_weight > 0: computes LM loss.
        Returns BaseRetrofitOutput with .loss, .grounding_loss, .anchor_positions.
        """
        output_hidden_states = True
        output_attentions = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ── Embed inputs ─────────────────────────────────────────────────────
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features ({n_image_features}) != image tokens "
                        f"({n_image_tokens}) in batch"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # ── RoPE position ids ─────────────────────────────────────────────────
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # ── Transformer forward ───────────────────────────────────────────────
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_position,
        )

        all_hidden_states = outputs.hidden_states
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        # ── Optional LM loss ─────────────────────────────────────────────────
        lm_loss = None
        if labels is not None and self.lm_loss_weight > 0:
            logits_f = logits.float()
            shift_logits = logits_f[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1).to(shift_logits.device),
            )

        # ── Grounding loss ────────────────────────────────────────────────────
        grounding_loss, all_grounding_scores, all_layer_weights, all_anchor_positions = (
            self._compute_grounding_loss(
                all_hidden_states=all_hidden_states,
                input_ids=input_ids,
                logits=logits,
                ground_token_indices=ground_token_indices,
                multi_patch_labels=multi_patch_labels,
                verbose=verbose,
            )
        )

        # ── Combine losses ────────────────────────────────────────────────────
        total_loss = None
        if lm_loss is not None and grounding_loss is not None:
            total_loss = self.lm_loss_weight * lm_loss + self.grounding_loss_weight * grounding_loss
        elif grounding_loss is not None:
            total_loss = self.grounding_loss_weight * grounding_loss
        elif lm_loss is not None:
            total_loss = lm_loss

        # ── Build output ─────────────────────────────────────────────────────
        if return_dict:
            return BaseRetrofitOutput(
                lm_loss=lm_loss,
                grounding_loss=grounding_loss,
                grounding_scores=all_grounding_scores,
                layer_weights=all_layer_weights,
                anchor_positions=(
                    all_anchor_positions if multi_patch_labels is not None else []
                ),
                loss=total_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=None,
                attentions=None,
                rope_deltas=self.rope_deltas,
            )
        else:
            if total_loss is not None:
                return (total_loss, logits) + outputs[1:]
            return (logits,) + outputs[1:]
