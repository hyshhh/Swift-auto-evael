"""用 gptqmodel 直接量化，绕过 optimum 的 TritonV2 内核问题

支持多模态模型（如 Qwen3.5 VL），同时量化语言模型和视觉编码器。
"""
import os, sys, json, shutil, torch
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from gptqmodel import GPTQModel
from gptqmodel.quantization.config import QuantizeConfig
from transformers import AutoTokenizer
from datasets import load_dataset

model_path = "/media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b"
output_path = "/media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-gptq"
dataset_path = "/media/ddc/新加卷/hys/qmy/agent/data/sft_val.jsonl"

# 量化配置 - 量化所有 Linear 层（包括 visual 编码器）
quantize_config = QuantizeConfig(
    bits=4,
    group_size=128,
    sym=True,
    desc_act=True,
    damp_percent=0.01,
    true_sequential=True,
    # 不排除任何层，确保 visual 编码器也被量化
    # modules_to_quantize=None 表示量化所有可量化的 Linear 层
)

# 加载模型
print("步骤 1: 加载模型...")
model = GPTQModel.load(
    model_path,
    quantize_config=quantize_config,
    trust_remote_code=True,
)

# 打印模型结构，确认 visual 编码器存在
print("\n模型结构：")
for name, module in model.model.named_children():
    param_count = sum(p.numel() for p in module.parameters())
    print(f"  {name}: {type(module).__name__} ({param_count / 1e6:.1f}M params)")

# 加载校准数据
print("\n步骤 2: 加载校准数据集...")
ds = load_dataset("json", data_files=dataset_path, split="train")
ds = ds.shuffle(seed=42).select(range(min(128, len(ds))))
calib_data = []
for item in ds:
    text = item.get("text", "") or (item.get("query", "") + " " + item.get("response", "")).strip()
    if text:
        calib_data.append(text)
print(f"加载 {len(calib_data)} 条校准数据")

# 执行量化
print("\n步骤 3: 执行量化（包括 visual 编码器）...")
model.quantize(calib_data, batch_size=1)

# 保存
print("\n步骤 4: 保存量化模型...")
try:
    model.save(output_path)
except AttributeError:
    model.save_quantized(output_path)

# 保存分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.save_pretrained(output_path)

# 复制所有配置文件（确保多模态配置完整）
print("步骤 5: 复制配置文件...")
output_dir = Path(output_path)
source_dir = Path(model_path)

config_files = [
    'config.json',
    'preprocessor_config.json',
    'tokenizer_config.json',
    'vocab.json',
    'merges.txt',
    'processor_config.json',
    'generation_config.json',
    'video_preprocessor_config.json',
    'chat_template.json',
    'special_tokens_map.json',
    'tokenizer.json',
]

for f in config_files:
    src = source_dir / f
    dst = output_dir / f
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print(f"  复制: {f}")

# 修改 config.json 中的 model_type，确保 vLLM 正确识别
config_path = output_dir / 'config.json'
if config_path.exists():
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 保留原始的 model_type（qwen3_5 或 qwen3_5_text）
    model_type = config.get('model_type', '')
    print(f"\n  model_type: {model_type}")

    # 确保 quantization_config 存在
    if 'quantization_config' not in config:
        config['quantization_config'] = {
            'quant_method': 'gptq',
            'bits': 4,
            'group_size': 128,
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print("  已添加 quantization_config")

print(f"\n量化完成！保存到: {output_path}")

# 列出输出目录的文件
print(f"\n输出目录文件：")
for f in sorted(output_dir.iterdir()):
    if f.is_file():
        size = f.stat().st_size
        if size > 1024 * 1024:
            print(f"  {f.name}: {size / 1024 / 1024:.1f} MB")
        else:
            print(f"  {f.name}: {size / 1024:.1f} KB")
