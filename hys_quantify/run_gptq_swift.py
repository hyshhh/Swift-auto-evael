"""用 gptqmodel 直接量化，绕过 optimum 的 TritonV2 内核问题"""
import os, sys, json, torch
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from gptqmodel import GPTQModel
from gptqmodel.quantization.config import QuantizeConfig
from transformers import AutoTokenizer
from datasets import load_dataset

model_path = "/media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b"
output_path = "/media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-gptq"
dataset_path = "/media/ddc/新加卷/hys/qmy/agent/data/sft_val.jsonl"

# 量化配置 - 使用 torch 后端避免 TritonV2 的 32 对齐问题
quantize_config = QuantizeConfig(
    bits=4,
    group_size=128,
    sym=True,
    desc_act=True,
    damp_percent=0.01,
    true_sequential=True,
)

# 加载模型
print("步骤 1: 加载模型...")
model = GPTQModel.load(
    model_path,
    quantize_config=quantize_config,
    trust_remote_code=True,
)

# 加载校准数据
print("步骤 2: 加载校准数据集...")
ds = load_dataset("json", data_files=dataset_path, split="train")
ds = ds.shuffle(seed=42).select(range(min(128, len(ds))))
calib_data = []
for item in ds:
    text = item.get("text", "") or (item.get("query", "") + " " + item.get("response", "")).strip()
    if text:
        calib_data.append(text)
print(f"加载 {len(calib_data)} 条校准数据")

# 执行量化
print("步骤 3: 执行量化...")
model.quantize(calib_data, batch_size=1)

# 保存
print("步骤 4: 保存量化模型...")
try:
    model.save(output_path)
except AttributeError:
    model.save_quantized(output_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.save_pretrained(output_path)

print(f"\n量化完成！保存到: {output_path}")
