import os
import cv2
import json
import types
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from copy import deepcopy

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from my_modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration,Qwen2_5_VLAttention

# ================= 配置区域 =================
PATCH_SIZE = 28
# 模型路径
MODEL_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/models/huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B" 
# 输入文件路径 (你的预测结果文件)
PRED_FILE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/look-ahead/eval/result/ByteDance-Seed/UI-TARS-1.5-7B/ScreenSpot-Pro.jsonl"
# Ground Truth 文件路径
GT_FILE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/look-ahead/data/ScreenSpot-Pro/uitars/eval.json"   # 假设是 JSONL 格式
# 结果保存路径
OUTPUT_DIR = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/look-ahead/attention_vis/results/uitars15/hotmap"
OUTPUT_JSON = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/look-ahead/attention_vis/results/uitars15/log.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# def enable_last_layer_attention_only(model):
#     """
#     修改模型，使其只在最后一层计算并返回 Attention Weights，
#     其他层强制使用 Flash Attention (如果不返回 weights，通常更省显存) 
#     或者返回 None 以节省显存。
#     """
    
#     # 获取所有层
#     layers = model.model.layers
#     num_layers = len(layers)
    
#     # 定义一个新的 forward 函数
#     # 注意：我们需要保留原始 forward 的所有逻辑，只是修改 output_attentions 的行为
    
#     # 策略：
#     # 我们不修改类方法，而是修改实例方法。
#     # 对于前 N-1 层，我们将它们的 forward 方法中的 output_attentions 参数强制设为 False。
#     # 对于最后一层，我们将 output_attentions 强制设为 True。
    
#     for i, layer in enumerate(layers):
#         # layer.self_attn 是 Qwen2_5_VLAttention 的实例
#         original_forward = layer.self_attn.forward
        
#         # 使用闭包捕获原始 forward 和层索引
#         def make_new_forward(original_f, layer_idx, is_last_layer):
#             def new_forward(self, hidden_states, attention_mask=None, *args, **kwargs):
#                 # 强制修改 output_attentions 参数
#                 # 只有最后一层才允许输出 attention
#                 kwargs['output_attentions'] = is_last_layer
                
#                 # 调用原始 forward
#                 return original_f(hidden_states, attention_mask, *args, **kwargs)
#             return new_forward
        
#         # 替换实例方法
#         is_last = (i == num_layers - 1)
#         # 绑定到实例上 (MethodType)
#         layer.self_attn.forward = types.MethodType(
#             make_new_forward(original_forward, i, is_last), 
#             layer.self_attn
#         )
        
#     print(f"Successfully patched {num_layers} layers. Only layer {num_layers-1} will output attentions.")

# ================= 使用方法 =================

# 1. 加载模型
# model = ...

# 2. 应用 Patch
# enable_last_layer_attention_only(model)

# 3. 推理
# 此时必须设置 output_attentions=True，否则 transformers 框架可能不会收集结果
# 但由于我们在底层拦截了，前 N-1 层实际上执行的是 output_attentions=False 的逻辑
# outputs = model(**inputs, output_attentions=True)

# 4. 获取结果
# outputs.attentions 将是一个 tuple
# 前 N-1 个元素可能是 None (取决于 transformers 如何处理返回值为 None 的情况)
# 或者 transformers 会报错？
# 让我们检查一下 transformers 的源码逻辑。


def load_json_or_jsonl(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.json'):
            data = json.load(f) 
        elif file_path.endswith('.jsonl'):
            data = []
            for line in f:
                data.append(json.loads(line))       
    return data

def load_data_maps(pred_path, gt_path):
    print("Loading data...")
    preds = load_json_or_jsonl(pred_path)
    gts = load_json_or_jsonl(gt_path)
    gt_map = {item['images'][0]:item for item in gts}
    return preds, gt_map

def normalize_bbox(bbox, width, height):
    """将绝对坐标 bbox 转换为 0-1000 的归一化坐标"""
    return [
        bbox[0] / width * 1000,
        bbox[1] / height * 1000,
        bbox[2] / width * 1000,
        bbox[3] / height * 1000
    ]

def is_point_in_bbox(x, y, bbox):
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]

def overlay_heatmap(image_path, attn_map, save_path, gt_bbox=None, pred_point=None):
    """
    将注意力热图叠加到原图上，并可选绘制 GT BBox 和预测点。
    
    Args:
        image_path (str): 原图路径
        attn_map (np.ndarray): 注意力权重矩阵
        save_path (str): 保存路径
        gt_bbox (list, optional): [x1, y1, x2, y2] 格式的 Ground Truth 边界框
        pred_point (tuple, optional): (x, y) 格式的预测点坐标
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    # 1. 处理热图
    # Resize attention map to image size
    heatmap = cv2.resize(attn_map, (img.shape[1], img.shape[0]))
    
    # Min-Max 归一化到 0-255
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_uint8 = np.uint8(255 * heatmap_norm)
    
    # 应用伪彩色 (蓝->红)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # 叠加热图 (原图 60% + 热图 40%)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    
    # 2. 绘制 GT BBox (如果提供)
    if gt_bbox is not None:
        # 确保坐标是整数
        x1, y1, x2, y2 = map(int, gt_bbox)
        # 绘制红色矩形框 (BGR: 0, 0, 255), 线宽 2
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # 可选：添加文字标签
        # cv2.putText(overlay, "GT", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 3. 绘制预测点 (如果提供)
    if pred_point is not None:
        px, py = map(int, pred_point)
        # 绘制显眼的绿色实心圆点 (BGR: 0, 255, 0)
        # 半径 5, 填充 (-1)
        cv2.circle(overlay, (px, py), 5, (0, 255, 0), -1)
        # 再画一个白色外圈增加对比度
        cv2.circle(overlay, (px, py), 6, (255, 255, 255), 1)

    # 4. 保存
    cv2.imwrite(save_path, overlay)
    # print(f"Saved visualization to {save_path}")

def process_sample(model, processor, pred_item, gt_item, sample_idx):
    output_json = deepcopy(gt_item)
    output_json['response'] = pred_item['response']
    # 1. 准备数据
    messages = pred_item["messages"]
    # 确保 images 格式适配 qwen_vl_utils
    # 输入格式中 images 是 [{"path": "..."}]，需要提取 path
    image_objs = pred_item.get("images", [])
    image_paths = [img["path"] if isinstance(img, dict) else img for img in image_objs]
    
    # 替换 messages 中的 <image> 占位符
    formatted_messages = []
    img_idx = 0
    for msg in messages:
        new_content = []
        if isinstance(msg["content"], str):
            parts = msg["content"].split("<image>")
            for i, part in enumerate(parts):
                if part:
                    new_content.append({"type": "text", "text": part})
                if i < len(parts) - 1 and img_idx < len(image_paths):
                    new_content.append({"type": "image", "image": image_paths[img_idx]})
                    img_idx += 1
        else:
            # 已经是 list 格式
            new_content = msg["content"]
        formatted_messages.append({"role": msg["role"], "content": new_content})

    # 2. 预处理
    text = processor.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(formatted_messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    # 3. 解析 Response 中的坐标
    # 找到 assistant 的回复部分
    # 简单起见，我们直接解析 input_ids，因为 input_ids 包含了完整的 prompt + response
    input_ids = inputs.input_ids[0]

    # 获取特殊 Token ID
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")
    box_start_id = processor.tokenizer.convert_tokens_to_ids("<|box_start|>")
    box_end_id = processor.tokenizer.convert_tokens_to_ids("<|box_end|>")
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    
    # 定位视觉区域
    try:
        input_ids_list = input_ids.tolist()
        v_start_idx = input_ids_list.index(vision_start_id)
        v_end_idx = input_ids_list.index(vision_end_id)
    except ValueError:
        print("Error: Vision tokens not found.")
        return False

    gt_bbox = gt_item["bbox"]
    img_w, img_h = gt_item["img_size"]
    grid_t, grid_h, grid_w = inputs.image_grid_thw[0].tolist()
    grid_h = (grid_h // 2)
    grid_w = (grid_w // 2)
    resized_img_w = grid_w * PATCH_SIZE
    resized_img_h = grid_h * PATCH_SIZE

    # 查找 Box
    # 我们只关心最后一部分（Assistant Response）里的 Box
    # 为了简化，我们遍历整个序列，取最后一个 Box 作为预测结果（通常只有一个）
    found_boxes = []
    input_len = len(input_ids_list)
    # 1. 从后向前寻找 box_end
    # range(start, stop, step) -> 从最后一个元素开始，倒序遍历到 0
    for end_pos in range(input_len - 1, -1, -1):
        if input_ids_list[end_pos] == assistant_id and input_ids_list[end_pos - 1] == im_start_id:
            response_start_idx = end_pos - 1
            break
        if input_ids_list[end_pos] == box_end_id:
            
            # 2. 找到 end 后，向前寻找最近的 start
            start_pos = -1
            for k in range(end_pos - 1, -1, -1):
                if input_ids_list[k] == box_start_id:
                    start_pos = k
                    break
                # 如果遇到另一个 end，说明结构嵌套或错误，停止当前查找（可选）
                if input_ids_list[k] == box_end_id:
                    break
            
            if start_pos != -1:
                # 提取内容: <|box_start|> [content] <|box_end|>
                content_ids = input_ids_list[start_pos+1 : end_pos]
                
                # 解析逻辑 (保持不变)
                box_str = processor.tokenizer.decode(content_ids)
                try:
                    clean_str = box_str.replace('(', '').replace(')', '')
                    parts = clean_str.split(',')
                    if len(parts) == 2:
                        pred_x = int(parts[0].strip())
                        pred_x = (pred_x / resized_img_w) * img_w
                        pred_y = int(parts[1].strip())
                        pred_y = (pred_y / resized_img_h) * img_h
                        
                        # 寻找逗号位置
                        comma_idx = -1
                        for idx, tid in enumerate(content_ids):
                            if ',' in processor.tokenizer.decode([tid]):
                                comma_idx = idx
                                break
                        
                        if comma_idx != -1:
                            base_idx = start_pos + 1
                            
                            # 辅助函数：判断是否为数字 token
                            def is_digit(tid):
                                return processor.tokenizer.decode([tid]).strip().isdigit()

                            x_indices = [base_idx + k for k in range(comma_idx) if is_digit(content_ids[k])]
                            y_indices = [base_idx + k for k in range(comma_idx+1, len(content_ids)) if is_digit(content_ids[k])]
                            
                            if x_indices and y_indices:
                                found_boxes.append({
                                    'x': int(pred_x), 'y': int(pred_y),
                                    'x_indices': x_indices, 'y_indices': y_indices
                                })
                    end_pos = start_pos
                except Exception as e:
                    # 解析失败，继续向前找下一个 box_end
                    raise

    if not found_boxes:
        print("No boxes found in response.")
        return False

    # 取最后一个 Box 作为预测结果
    pred_box = found_boxes[-1]

    # 4. Forward Pass (Teacher Forcing)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=False, response_start_idx=response_start_idx)
    
    # 5. 验证准确率
    # gt_bbox_norm = normalize_bbox(gt_bbox, img_w, img_h)
    gt_bbox_norm = gt_bbox
    
    is_correct = is_point_in_bbox(pred_box['x'], pred_box['y'], gt_bbox_norm)
    output_json['is_correct'] = is_correct
    # print(f"Sample {sample_idx}: Pred({pred_box['x']}, {pred_box['y']}) vs GT_Norm({[int(x) for x in gt_bbox_norm]}) -> {'Correct' if is_correct else 'Wrong'}")

    # 6. 可视化注意力
    last_layer_attn = outputs.attentions[-1] # (batch, heads, seq, seq)
    attn_matrix = last_layer_attn[0].mean(dim=0).float().cpu() # (seq, seq)
    
    # 校验形状
    visual_len = v_end_idx - (v_start_idx + 1)
    img_path = image_paths[0]
    img_name = os.path.basename(img_path).split('.')[0]
    output_dir = os.path.join(OUTPUT_DIR, img_name)
    # 创建子目录
    x_dir = os.path.join(output_dir, "x")
    y_dir = os.path.join(output_dir, "y")
    os.makedirs(x_dir, exist_ok=True)
    os.makedirs(y_dir, exist_ok=True)

    # 用于计算最终的总平均
    all_tokens_attn_sum = torch.zeros(visual_len, device='cpu')
    total_token_count = 0

    for axis in ['x', 'y']:
        indices = pred_box[f'{axis}_indices']
        if not indices: continue
        
        axis_dir = x_dir if axis == 'x' else y_dir
        axis_attn_sum = torch.zeros(visual_len, device='cpu')
        
        # 遍历该分量的每一个 Token
        for i, token_idx in enumerate(indices):
            # 获取 Token 的文本值 (用于文件名)
            token_val = processor.tokenizer.decode([input_ids_list[token_idx]]).strip()
            
            # 提取单 Token 注意力
            raw_attn = attn_matrix[token_idx - response_start_idx, v_start_idx+1 : v_end_idx] # (visual_len,)
            
            # 累加到 Axis 总和
            axis_attn_sum += raw_attn
            # 累加到 Global 总和
            all_tokens_attn_sum += raw_attn
            total_token_count += 1
            
            # === 单 Token 热图 ===
            # 归一化
            attn_sum = raw_attn.sum()
            if attn_sum > 0:
                norm_attn = raw_attn / attn_sum
                try:
                    attn_map = norm_attn.view(grid_h, grid_w).numpy()
                    save_name = f"{axis}_token_{i}_val_{token_val}.png"
                    overlay_heatmap(img_path, attn_map, os.path.join(axis_dir, save_name), gt_bbox, (pred_box['x'], pred_box['y']))
                except:
                    raise

        # === Axis 平均热图 (X_avg 或 Y_avg) ===
        if len(indices) > 0:
            # 计算平均：总和 / token数量
            avg_axis_attn = axis_attn_sum / len(indices)
            
            # 归一化
            attn_sum = avg_axis_attn.sum()
            if attn_sum > 0:
                avg_axis_attn = avg_axis_attn / attn_sum
                
            try:
                attn_map = avg_axis_attn.view(grid_h, grid_w).numpy()
                save_name = f"{axis}_avg.png"
                overlay_heatmap(img_path, attn_map, os.path.join(axis_dir, save_name), gt_bbox, (pred_box['x'], pred_box['y']))
            except:
                raise

    # === Global Combined 热图 (X和Y所有Token的平均) ===
    if total_token_count > 0:
        # 计算总平均
        global_avg_attn = all_tokens_attn_sum / total_token_count
        
        # 归一化
        attn_sum = global_avg_attn.sum()
        output_json['total_vision_attention_weights'] = attn_sum.item() # 记录总权重值
        
        if attn_sum > 0:
            global_avg_attn = global_avg_attn / attn_sum
            
        try:
            attn_map = global_avg_attn.view(grid_h, grid_w).numpy()
            # 保存到最外层
            save_name = f"combined_xy_{'correct' if is_correct else 'wrong'}.png"
            overlay_heatmap(img_path, attn_map, os.path.join(output_dir, save_name), gt_bbox, (pred_box['x'], pred_box['y']))
            print(f"Saved combined heatmap to {save_name}")
        except:
            raise
            
    return output_json

def main():
    # 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda"
    )
    # enable_last_layer_attention_only(model)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    output_jsons = []
    # 加载数据
    preds, gt_map = load_data_maps(PRED_FILE, GT_FILE)
    print(f"Loaded {len(preds)} predictions and {len(gt_map)} GT items.")

    correct_count = 0
    total_count = 0

    for i, pred_item in tqdm(enumerate(preds)):
        img_path = pred_item["images"][0]["path"]
        gt_item = gt_map[img_path]
        try:
            output_json = process_sample(model, processor, pred_item, gt_item, i)
            if output_json['is_correct']:
                correct_count += 1
            total_count += 1
            output_jsons.append(output_json)
            with open(OUTPUT_JSON, mode='w', encoding='utf-8') as f:
                json.dump(output_jsons, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()

    if total_count > 0:
        print(f"\nFinal Accuracy: {correct_count}/{total_count} = {correct_count/total_count:.2%}")
    else:
        print("\nNo valid samples processed.")


if __name__ == "__main__":
    main()