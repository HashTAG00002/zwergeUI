import os
import cv2
import json
import types
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import AutoProcessor

from grounding_attention import process_sample, load_data_maps
from my_modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

# ================= 配置区域 =================
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

# ================= 自定义 Dataset =================
class InferenceDataset(Dataset):
    def __init__(self, preds, gt_map):
        self.data = []
        for i, pred_item in enumerate(preds):
            img_path = pred_item["images"][0]["path"]
            # 查找 GT
            gt_item = gt_map.get(img_path)
            if not gt_item:
                # 尝试文件名匹配
                img_name = os.path.basename(img_path)
                for k, v in gt_map.items():
                    if os.path.basename(k) == img_name:
                        gt_item = v
                        break
            
            if gt_item:
                self.data.append({
                    "index": i,
                    "pred_item": pred_item,
                    "gt_item": gt_item
                })
            else:
                print(f"Warning: No GT for {img_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 这里返回原始数据即可，复杂的预处理（如 processor）建议放在 collate_fn 或处理循环中
        # 因为 processor 输出的是 Tensor，放在 Dataset 里可能会导致多进程 pickle 问题或显存增加
        return self.data[idx]

# ================= Collate Function =================
def collate_fn(batch):
    # 简单地将 list of dicts 转换为 dict of lists，方便后续处理
    return batch


def main():
    # 1. 初始化 Accelerator
    # 它会自动检测环境（单卡/多卡/DeepSpeed等）
    accelerator = Accelerator()
    
    # 仅在主进程打印日志
    if accelerator.is_main_process:
        print(f"Loading model on {accelerator.device}...")

    # 2. 加载模型和处理器
    # 注意：不要指定 device_map="cuda"，accelerator 会自动管理 device
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # 如果有 Patch，在这里应用
    # enable_last_layer_attention_only(model)

    # 3. 准备数据
    if accelerator.is_main_process:
        print("Loading data...")
        preds, gt_map = load_data_maps(PRED_FILE, GT_FILE)
    else:
        preds, gt_map = None, None
    
    # 广播数据给所有进程（或者让每个进程都加载一遍，如果数据不大）
    # 简单起见，这里假设每个进程都加载一遍数据（几万条数据内存占用不大）
    if preds is None: 
        preds, gt_map = load_data_maps(PRED_FILE, GT_FILE)

    dataset = InferenceDataset(preds, gt_map)
    
    # DataLoader
    # batch_size 可以设大一点，但对于 LLM 推理，通常设为 1 比较安全（避免 padding 浪费和 OOM）
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2 # 可以开启多进程加载数据
    )

    # 4. Prepare with Accelerator
    # 这一步非常关键！它会自动：
    # - 将模型移动到正确的 GPU
    # - 将 DataLoader 切分给不同的 GPU (DistributedSampler)
    model, dataloader = accelerator.prepare(model, dataloader)

    model.eval()
    
    local_results = []
    
    # 5. 推理循环
    # disable=not accelerator.is_local_main_process 确保只有主进程显示进度条
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        # batch 是一个 list (因为 batch_size=1 且用了自定义 collate)，里面包含 1 个样本
        # 或者如果 batch_size > 1，就是 list of dicts
        
        for item in batch:
            idx = item['index']
            pred_item = item['pred_item']
            gt_item = item['gt_item']
            
            try:
                # 调用你的处理函数
                # 注意：process_sample 内部需要使用 model.device，现在它已经是正确的 device 了
                # 这里的 model 可能是被 DDP 包装过的，访问原始方法可能需要 model.module
                unwrapped_model = accelerator.unwrap_model(model)
                
                output_json = process_sample(unwrapped_model, processor, pred_item, gt_item, idx)
                local_results.append(output_json)
                
            except Exception as e:
                print(f"Error on sample {idx}: {e}")

    # 6. 收集结果 (Gather)
    # 这一步稍微麻烦点，因为 json 对象很难直接 gather。
    # 推荐做法：每个进程保存自己的结果到文件，最后主进程合并。
    # 或者使用 accelerator.gather_for_metrics (如果结果是 tensor)
    
    # 最稳妥方案：每个进程写自己的文件
    output_part = f"{OUTPUT_JSON}.part{accelerator.process_index}"
    with open(output_part, 'w', encoding='utf-8') as f:
        json.dump(local_results, f, indent=2, ensure_ascii=False)
    
    # 等待所有进程写完
    accelerator.wait_for_everyone()
    
    # 7. 主进程合并结果
    if accelerator.is_main_process:
        print("Merging results...")
        final_results = []
        total_correct = 0
        total_count = 0
        
        for i in range(accelerator.num_processes):
            part_file = f"{OUTPUT_JSON}.part{i}"
            if os.path.exists(part_file):
                with open(part_file, 'r') as f:
                    chunk = json.load(f)
                    final_results.extend(chunk)
                    for res in chunk:
                        if res.get('is_correct'):
                            total_correct += 1
                        total_count += 1
                os.remove(part_file)
        
        # 保存最终结果
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
            
        print(f"Final Accuracy: {total_correct}/{total_count} = {total_correct/total_count:.2%}")


if __name__ == "__main__":
    main()