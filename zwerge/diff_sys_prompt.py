"""比对训练时 system_message 和当前 constants.py 的差异"""
import json, sys, difflib
sys.path.insert(0, "src")

ARGS_JSON = (
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03"
    "/.hdd/ckpt/zwerge/guiowl_grounding50k_A3-gaussian_cos_meta_20260521_210923/args.json"
)

with open(ARGS_JSON) as f:
    args = json.load(f)
train_sys = args["data_args"]["system_message"]
train_grd = args["data_args"]["ground_response"]

from zwerge_retrofit.constants import GUI_OWL_SYSTEM_PROMPT as current_sys, GUI_OWL_GROUND_RESPONSE as current_grd

print(f"=== system_message ===")
print(f"train  len={len(train_sys)}")
print(f"current len={len(current_sys)}")
print(f"identical: {train_sys == current_sys}")
if train_sys != current_sys:
    diff = list(difflib.unified_diff(
        train_sys.splitlines(keepends=True),
        current_sys.splitlines(keepends=True),
        fromfile="train_sys", tofile="current_sys"
    ))
    print("\n--- diff (first 60 lines) ---")
    for line in diff[:60]:
        print(repr(line))

print(f"\n=== ground_response ===")
print(f"train  : {repr(train_grd)}")
print(f"current: {repr(current_grd)}")
print(f"identical: {train_grd == current_grd}")
