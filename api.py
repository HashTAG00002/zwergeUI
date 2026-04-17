"""
Adapted from /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/jinchao10/gpt.py
"""

import cv2
import multiprocessing
import os
import json
import base64
import re
import signal
import time
from PIL import Image
from io import BytesIO
from flask.cli import F
from tqdm import tqdm
from octo_rpc import NonMeshClient
from octo_rpc import load
import numpy as np

# <thinking>
# ...
# ROI: <|object_ref_start|>...<|object_ref_end|>
# <|box_start|>(x1,y1),(x2,y2)<|box_end|>
# ...
# UI: <|object_ref_start|><|object_ref_end|>
# </thinking>
# <answer><|box_start|>(x1,y1),(x2,y2)<|box_end|></answer>
QA_SYSTEM_PROMPT = """
You are an expert Grounding GUI Agent. Your task is to locate the exact bounding box (bbox) of the UI element for interaction based on visual screenshots and instructions.

### OUTPUT FORMAT
Your response MUST strictly follow this structure with ONLY necessary concise thinking:
<thinking>
{your thinking about ROI}... ROI: <|object_ref_start|>{ROI description}<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
{your thinking about UI}... UI: <|object_ref_start|>{UI description}<|object_ref_end|>
</thinking>
<answer><|box_start|>(x1,y1),(x2,y2)<|box_end|></answer>


### REQUIREMENTS
1. **Region of Interest (ROI)**: 
   - Define a semantically complete rectangular area (e.g., a menu, a button group, or a specific card).
   - Elements within an ROI should be spatially proximate and functionally parallel.
   - The ROI should have clear boundaries and be mutually exclusive from other ROIs.
2. **Specific UI Element (UI)**: 
   - Focus exclusively on the visual tokens inside the previously defined ROI.
   - Ensure the UI bbox is strictly contained within the ROI bbox.
3. **Coordinates**: 
   - All pixel coordinates in (x1,y1),(x2,y2) format are normalized to [0,1000].
Note: No conversational filler. Strictly follow the tags."""

QA_USER_PROMPT = """
Current Instruction: {step_instruction}
Screenshot: 
""".strip()

def extract_and_validate(content, gt_bbox):
    try:
        # 1. 提取核心标签内容
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", content, re.S | re.I)
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.S | re.I)
        
        if not thinking_match or not answer_match:
            return False, None
            
        thinking = thinking_match.group(1).strip()
        answer = answer_match.group(1).strip()

        # 2. 提取 ROI 相关信息
        # 匹配格式: [thinking about ROI]... ROI: <|object_ref_start|>region_description<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
        roi_pattern = r"(.*?)ROI:\s*<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>"
        roi_res = re.search(roi_pattern, thinking, re.S)
        
        thinking_about_roi = roi_res.group(1).strip()
        region_description = roi_res.group(2).strip()
        roi = [int(x) for x in roi_res.groups()[2:]] # [x1, y1, x2, y2]

        # 3. 提取 UI 相关信息
        # 匹配格式: [thinking about UI]... UI: <|object_ref_start|>element_description<|object_ref_end|>
        # 注意：UI 的坐标通常在 <answer> 标签中，或者在 thinking 的末尾
        ui_pattern = r"(.*?)UI:\s*<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
        ui_res = re.search(ui_pattern, thinking, re.S)
        
        thinking_about_ui = ui_res.group(1).replace(roi_res.group(0), "").strip() # 剔除掉ROI部分的内容
        element_description = ui_res.group(2).strip()

        # 4. 提取 Answer 坐标
        ans_box_str = re.search(r"<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>", answer).groups()
        ui_bbox = [int(x) for x in ans_box_str]

        info = {
            "roi": roi,
            "ui": ui_bbox,
            "thinking_about_roi": thinking_about_roi,
            "region_description": region_description,
            "thinking_about_ui": thinking_about_ui,
            "element_description": element_description,
        }
        # --- Rejection 采样逻辑 ---

        # 检查 1: UI bbox 是否完全落在 ROI 区域中
        if not (roi[0] <= ui_bbox[0] and roi[1] <= ui_bbox[1] and roi[2] >= ui_bbox[2] and roi[3] >= ui_bbox[3]):
            return False, info
            
        # 检查 2: UI bbox 的中心点是否落在 Ground Truth (gt_bbox) 中
        ui_center_x = (ui_bbox[0] + ui_bbox[2]) / 2
        ui_center_y = (ui_bbox[1] + ui_bbox[3]) / 2
        if gt_bbox[0] <= ui_center_x <= gt_bbox[2] and gt_bbox[1] <= ui_center_y <= gt_bbox[3]:
            success = True
        else:
            success = False
            
        # 返回所有提取到的字段
        return success, info
    except Exception as e:
        # print(f"Extract failed: {e}")
        return False, None
    


current_path = os.getcwd()
gptThrift = load(os.path.join(current_path, "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/jinchao10/Hallucination/cotGeneration/APIGpt.thrift"))

# 全局变量存储结果
results = []
last_save_time = 0
SAVE_INTERVAL = 60  # 保存间隔（秒）
# model_name = "gpt-4o-2024-08-06"
# model_name = "gpt-4.1"
# model_name = "gemini-2.5-pro"
# model_name = "vertex.claude-sonnet-4"
# model_name = "gemini-2.5-pro"
# model_name = "qwen3-vl-plus"
# model_name = "doubao-seed-1-6-vision-250815"
# model_name = "LongCat-Flash-Omni"
model_name = "gemini-3-flash-preview"

def overlay_heatmap(image_path, attn_map, save_path, gt_bbox=None, pred_point=None, normalized=True):
    img = cv2.imread(image_path)
    if img is None: return
    heatmap = cv2.resize(attn_map, (img.shape[1], img.shape[0]))
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_uint8 = np.uint8(255 * heatmap_norm)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    if gt_bbox is not None:
        x1, y1, x2, y2 = map(int, gt_bbox)
        if normalized:
            x1 = int(x1 / 1000 * img.shape[1])
            x2 = int(x2 / 1000 * img.shape[1])
            y1 = int(y1 / 1000 * img.shape[0])
            y2 = int(y2 / 1000 * img.shape[0])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if pred_point is not None:
        px, py = map(int, pred_point)
        if normalized:
            px = int(px / 1000 * img.shape[1])
            py = int(py / 1000 * img.shape[0])
        cv2.circle(overlay, (px, py), 5, (0, 255, 0), -1)
        cv2.circle(overlay, (px, py), 6, (255, 255, 255), 1)
    cv2.imwrite(save_path, overlay)

def create_mesh_client():
    client = NonMeshClient(
        service=gptThrift.APIGptService,
        service_name="com.sankuai.horus.hugging.adapter.server.APIGptService",
        appkey="com.sankuai.wmocr.charmattnnlp",
        remote_appkey="com.sankuai.horus.hugging.adapter",
        filter_by_service_name=True,
        timeout=5000000
    )
    print("Client created successfully")
    return client

def save_results(output_file, force=False):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    global last_save_time
    current_time = time.time()
    if force or (current_time - last_save_time > SAVE_INTERVAL):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_file} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        last_save_time = current_time

def make_handler(output_file):
    def _handler(sig, frame):
        print("Program interrupted. Saving current results...")
        save_results(output_file, force=True)
        print("Results saved. Exiting...")
        exit(0)
    return _handler

def process_chunk(screenshot_dir, output_dir, chunk, chunk_index, shared_results):
    """
    新增参数: shared_results (multiprocessing.Manager().list())
    """
    client = create_mesh_client()
    
    for data in tqdm(chunk, desc=f"Chunk {chunk_index}"):
        try:
            image_path = data['images'][0]
            if not image_path: continue
            
            # ... [中间的图片处理和大模型请求逻辑保持不变] ...
            with Image.open(image_path).convert("RGB") as im_pil:
                w, h = im_pil.size
                buffered = BytesIO()
                im_pil.save(buffered, format="JPEG")
                im_pil_stream = base64.b64encode(buffered.getvalue()).decode('utf-8')

            user_prompt = QA_USER_PROMPT.format(step_instruction=data["instruction"])
            org_req = {
                "model": model_name, 
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": QA_SYSTEM_PROMPT}]},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt}, 
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{im_pil_stream}"}}
                        ]
                    }
                ]
            }

            req = gptThrift.GetGptRawReq(json=json.dumps(org_req, ensure_ascii=False))
            resp_data = json.loads(client.getGptRaw(req).data)
            content = resp_data["choices"][0]["message"]["content"]
            print(content)
            
            # 1. 执行校验
            success, info = extract_and_validate(content, data["normalized_bbox"])
            # 2. 将 info 写入 data 数据项
            data["rejection_sampling_info"] = info
            data["model_raw_content"] = content
            data["roi_success"] = success
            
            # 3. 绘制热力图逻辑
            attn_map = np.zeros((1000, 1000), dtype=np.float32)
            r, u = info["roi"], info["ui"]
            attn_map[r[1]:r[3], r[0]:r[2]] = 0.5 
            attn_map[u[1]:u[3], u[0]:u[2]] = 1.0 
            
            save_name = image_path.replace(screenshot_dir, output_dir, 1)
            data["roi_image"] = save_name
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            overlay_heatmap(image_path, attn_map, save_name, 
                            gt_bbox=data["normalized_bbox"], pred_point=((u[0]+u[2])/2, (u[1]+u[3])/2))
            
            # 4. 安全地写入共享列表
            shared_results.append(data)
                
        except Exception as e:
            print(f"Error in chunk {chunk_index}: {e}")
            continue


def main():
    input_json = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/attention_vis/data/generate_cot/ScreenSpot-Pro/ori_eval.json"
    output_json = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/attention_vis/data/generate_cot/ScreenSpot-Pro/Seed/roi.json"
    screenshot_dir = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/ScreenSpot-Pro/images"
    base_output_dir = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/attention_vis/data/generate_cot/ScreenSpot-Pro/Seed/vis_hotmap"
    all_data = []

    # 遍历所有 json 文件
    # for json_file in os.listdir(anno_dir):
    #     if not json_file.endswith(".json"): continue
        
    #     json_path = os.path.join(anno_dir, json_file)
    #     with open(json_path, 'r', encoding='utf-8') as f:
    #         data_list = json.load(f)
        
    #     all_data.extend(data_list)
    with open(input_json, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    # --- 多进程安全组件 ---
    manager = multiprocessing.Manager()
    shared_results = manager.list() # 创建跨进程共享列表

    num_chunks = 5 # 建议增加块数以发挥多核优势
    chunks = np.array_split(all_data, min(num_chunks, len(all_data)))

    if len(chunks) == 1:
        process_chunk(screenshot_dir, base_output_dir, chunks[0].tolist(), 0, shared_results)
    else:
        processes = []
        for i, chunk in enumerate(chunks):
            p = multiprocessing.Process(
                target=process_chunk, 
                args=(screenshot_dir, base_output_dir, chunk.tolist(), i, shared_results)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    # --- 所有进程结束后，由主进程统一保存 ---
    print(f"\n>>> All processes finished. Saving {len(shared_results)} successful items to {output_json}")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        # 将共享列表转换为普通列表并保存
        json.dump(list(shared_results), f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()