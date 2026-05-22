"""检查 guiowl tokenizer 的 special token IDs"""
import sys
sys.path.insert(0, "src")

from transformers import AutoTokenizer

CKPT = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/guiowl_grounding50k_A3-gaussian_cos_meta_20260521_210923/checkpoint-800"
tok = AutoTokenizer.from_pretrained(CKPT)

special = ["<|ground|>", "<|pointer_start|>", "<|pointer_end|>", "<|pointer_pad|>"]
for s in special:
    tid = tok.convert_tokens_to_ids(s)
    print(f"{s!r:30} -> {tid}")

print()
from zwerge_retrofit.constants import GUI_OWL_GROUND_RESPONSE, GUI_OWL_SYSTEM_PROMPT
ids = tok.encode(GUI_OWL_GROUND_RESPONSE, add_special_tokens=False)
print("GUI_OWL_GROUND_RESPONSE:")
print(repr(GUI_OWL_GROUND_RESPONSE))
print("encoded:", ids)
for i, tid in enumerate(ids):
    print(f"  [{i}] {tid}: {repr(tok.decode([tid]))}")

print()
print("First 20 tokens of system prompt:")
sys_ids = tok.encode(GUI_OWL_SYSTEM_PROMPT, add_special_tokens=False)
print(f"  system prompt len (tokens): {len(sys_ids)}")
# Find ground token positions
import json
gt_id = tok.convert_tokens_to_ids("<|ground|>")
positions = [i for i, t in enumerate(sys_ids) if t == gt_id]
print(f"  <|ground|> positions in system: {positions}")
for pos in positions:
    ctx = sys_ids[max(0,pos-3):pos+5]
    decoded = [repr(tok.decode([t])) for t in ctx]
    print(f"    pos {pos}: {decoded}")

print()
print("Check added_tokens from checkpoint:")
import json
with open(f"{CKPT}/added_tokens.json") as f:
    added = json.load(f)
print(json.dumps(added, indent=2, ensure_ascii=False))
