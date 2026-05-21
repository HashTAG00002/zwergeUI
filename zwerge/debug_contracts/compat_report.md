# Compatibility Audit: GUIOwlRetrofitModel / Qwen3-VL Retrofit

**Date:** 2026-05-21

---

## Environment Versions

| Component | qwen3-verl (GUI-Owl) | gui_actor (UI-TARS) |
|---|---|---|
| transformers | **4.57.1** | 4.51.3 |
| torch | 2.9.1+cu128 | 2.5.1 |
| accelerate | 1.6.0 | 1.1.1 |
| `check_model_inputs` decorator | EXISTS, wraps Qwen3VL forwards | **DOES NOT EXIST** |
| `_CAN_RECORD_REGISTRY` | EXISTS, **EMPTY** | ImportError |
| `use_reentrant` default | **True** | True |

---

## Forward Signatures (qwen3-verl 4.57.1)

### Qwen3VLModel.get_rope_index
```
(self, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None)
```
→ **NO `mm_token_type_ids`** (GPT's advice about this param was for a different version)

### Qwen3VLModel.get_placeholder_mask
```
(self, input_ids, inputs_embeds, image_features=None, video_features=None)
```
→ use keyword args; returns `(image_mask, video_mask)` tuple

### Qwen3VLModel.get_image_features
```
(self, pixel_values, image_grid_thw=None) → (image_embeds_list, deepstack_image_embeds_list)
```

### Qwen3VLTextModel.forward (the INNER language model)
Accepts `visual_pos_masks`, `deepstack_visual_embeds` ✓. Returns `hidden_states` when `output_hidden_states=True` is passed via `**kwargs`.

### Qwen3VLModel.forward / Qwen3VLForConditionalGeneration.forward
**CRITICAL BUG IN QWEN3VL:** `Qwen3VLModel.forward` passes `output_hidden_states` down to `Qwen3VLTextModel` via `**kwargs`, but then constructs its return as:
```python
return Qwen3VLModelOutputWithPast(last_hidden_state=..., past_key_values=..., rope_deltas=...)
# hidden_states NOT included even when output_hidden_states=True
```
→ Any code using `Qwen3VLForConditionalGeneration.forward(output_hidden_states=True)` gets `base_output.hidden_states = None`.

---

## API Assumptions: Verified vs Broken

| Assumption | Status |
|---|---|
| `self.model.language_model.embed_tokens` exists | ✓ VERIFIED |
| `get_image_features` returns `(list, list)` | ✓ VERIFIED |
| `get_placeholder_mask` with keyword args | ✓ VERIFIED |
| `get_rope_index(input_ids, thw, None, attn_mask)` | ✓ VERIFIED |
| `language_model` accepts `visual_pos_masks`, `deepstack_visual_embeds` | ✓ VERIFIED |
| `output_hidden_states=True` via outer Qwen3VL wrapper returns hidden states | ✗ BROKEN — silently returns None |
| `check_model_inputs` monkey-patches 36 layers | ✗ FALSE — registry empty, no-op |

---

## Trainable Parameters

| Parameter | requires_grad | Gets gradient? |
|---|---|---|
| `layerwise_grounding_head.*` (~12M) | True | ✓ via grounding loss |
| `_new_token_emb` [4, d_model] | True | ⚠ only if batch contains special tokens |
| backbone params | False | — |

---

## DDP Configuration

HF Trainer sets:
```python
find_unused_parameters = (
    training_args.ddp_find_unused_parameters   # if not None
    else not training_args.gradient_checkpointing  # implicit
)
```
With `gradient_checkpointing=True` (from scripts) and `ddp_find_unused_parameters=None`:
→ `find_unused_parameters = False`
→ DDP fails when `_new_token_emb` or any skip branch misses gradient.

---

## Root Cause of All Issues

### Why GUI-Owl broke (Qwen3-VL, transformers 4.57.1):
1. `gradient_checkpointing=True` → `find_unused_parameters=False` → DDP `_rebuild_buckets` crash
2. Skip branches used `logits[i].sum() * 0.0` (NOT connected to grounding head) → DDP inconsistency
3. `Qwen3VLModel.forward` silently drops `hidden_states` → any old `output_hidden_states=True` path fails

### Why UI-TARS works (Qwen2.5-VL, transformers 4.51.3):
1. `check_model_inputs` decorator doesn't exist → no wrapper overhead at all
2. Older PyTorch/accelerate behavior is more lenient with unused params
3. `Qwen2_5_VLModel.forward` correctly propagates `hidden_states` when requested
4. No deepstack, simpler computation graph

---

## Fixes Applied

### P0: `train_retrofit.py`
- Disable `gradient_checkpointing` for guiowl/uivenus (not needed for frozen backbone)
- Explicitly set `ddp_find_unused_parameters=True` (not relying on Trainer's implicit logic)

### P0: `modeling_base.py`
- Add `_zero_grounding_loss()` helper connected to all trainable params
- Replace all `logits[i].sum() * 0.0` skip-branch losses with `_zero_grounding_loss()`

### P0: `modeling_guiowl.py` (already done)
- Hook-based `_run_language_model` bypasses `Qwen3VLModel.forward` entirely
- `output_hidden_states=False` — no hidden_states overhead at all
- Only captures 15 probe layer outputs via `.detach()` hooks

### P1: `modeling_guiowl.py`
- Added hook shape/type assertion

---

## `_new_token_emb` Semantic Note

With the hook-based approach and `.detach()`:
- The grounding loss does NOT backprop through `_new_token_emb`
- `_new_token_emb` only gets gradients if the dataset uses a different path

**Recommendation:** For the "internal-signal" paper claim (grounding emerges before decoding), treat this as **head-only retrofit**. `_new_token_emb` is a tokenizer compatibility placeholder; its gradient is irrelevant. `find_unused_parameters=True` handles this cleanly.

---

## Verification Scripts

- `debug_contracts/inspect_qwen3vl.py` — verifies all API assumptions
- `debug_contracts/test_ddp_edge_batches.py` — verifies gradient connectivity

Run in qwen3-verl env:
```bash
cd zwerge
python debug_contracts/inspect_qwen3vl.py
python debug_contracts/test_ddp_edge_batches.py
```
