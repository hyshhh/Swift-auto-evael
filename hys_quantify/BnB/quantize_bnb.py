#!/usr/bin/env python3
"""
BnB NF4 量化脚本 - 适配 Qwen3-VL 多模态模型
量化语言模型部分，视觉编码器保持 FP16/BF16

支持 vLLM 部署（INT4 量化）:
    vllm serve /path/to/model --quantization bitsandbytes --load-format bitsandbytes

参考: https://docs.vllm.com.cn/en/latest/features/quantization/int4/

Qwen3-VL 模型架构:
    Qwen3VLForConditionalGeneration
    ├── model (Qwen3VLModel)
    │   ├── embed_tokens
    │   ├── layers (x N)
    │   ├── norm
    │   └── visual (Qwen3VisionTransformer) ← 视觉编码器在这里
    └── lm_head

    注意: named_children() 只返回 model 和 lm_head，看不到 model.visual
          必须用 named_modules() 递归搜索才能找到视觉编码器
"""

import os
import gc
import json
import argparse
import shutil
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)


def detect_model_type(model_path):
    """检测模型类型，返回正确的加载类名"""
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        model_type = config.get('model_type', '')

        # 检查是否为 Qwen3 VL 多模态模型（qwen3_vl 而不是 qwen3_5）
        if model_type == 'qwen3_vl':
            return 'qwen3_vl'
        # 检查是否为 Qwen2 VL 多模态模型
        elif model_type == 'qwen2_vl':
            return 'qwen2_vl'
        # 检查是否有 visual_config（通用多模态检测）
        elif 'visual_config' in config or 'vision_config' in config:
            return 'multimodal'
        # Qwen3.5 纯文本模型
        elif model_type == 'qwen3_5':
            return 'qwen3_5_text'
        # 其他 Qwen 纯文本模型
        elif 'qwen' in model_type:
            return 'qwen_text'

    return 'text_only'


def copy_config_files(official_model, output_path):
    """从官方模型复制配置文件，确保 vLLM 兼容性"""
    official_model = Path(official_model)
    output_path = Path(output_path)

    config_files = [
        'config.json',
        'video_preprocessor_config.json',
        'image_preprocessor_config.json',
        'preprocessor_config.json',
        'generation_config.json',
        'tokenizer_config.json',
        'tokenizer.json',
        'vocab.json',
        'merges.txt',
        'special_tokens_map.json',
        'chat_template.json',
        'processor_config.json',
    ]

    for config_file in config_files:
        src = official_model / config_file
        dst = output_path / config_file
        if src.exists():
            shutil.copy2(src, dst)
            print(f'  复制: {config_file}')


def patch_config_for_vllm(output_path, quant_bits, quant_type):
    """修补 config.json，添加 vLLM 需要的 quantization_config 字段"""
    config_path = Path(output_path) / "config.json"
    if not config_path.exists():
        print("  ⚠ config.json 不存在，跳过修补")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 添加 quantization_config，vLLM 靠此字段自动识别量化方式
    config['quantization_config'] = {
        "quant_method": "bitsandbytes",
        "bits": quant_bits,
        "bnb_4bit_quant_type": quant_type,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
        "load_in_4bit": quant_bits == 4,
        "load_in_8bit": quant_bits == 8,
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("  ✓ 已修补 config.json（添加 quantization_config）")


def parse_args():
    parser = argparse.ArgumentParser(description="BnB NF4 量化 Qwen3-VL 多模态模型")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--output", type=str, required=True, help="输出路径")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="量化位数 (4=NF4, 8=INT8)")
    parser.add_argument("--double_quant", action="store_true", help="启用双重量化（进一步压缩）")
    parser.add_argument("--compute_dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"], help="计算精度")
    parser.add_argument("--max_shard_size", type=str, default="2GB", help="每个分片最大大小")
    parser.add_argument("--use_safetensors", action="store_true", default=True, help="使用 safetensors 格式（推荐）")
    parser.add_argument("--copy_config", action="store_true", help="从官方模型复制配置文件")
    parser.add_argument("--official_model", type=str, default=None, help="官方模型路径（用于复制配置）")
    parser.add_argument("--push_to_hub", type=str, default=None, help="推送到 HuggingFace Hub")
    return parser.parse_args()


def get_compute_dtype(dtype_str: str):
    """获取计算精度"""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_str]


def main():
    args = parse_args()

    print("=" * 60)
    print("BnB NF4 量化 - Qwen3-VL 多模态模型")
    print("=" * 60)
    print(f"模型路径: {args.model}")
    print(f"输出路径: {args.output}")
    print(f"量化位数: {args.bits}-bit {'NF4' if args.bits == 4 else 'INT8'}")
    print(f"双重量化: {'是' if args.double_quant else '否'}")
    print(f"计算精度: {args.compute_dtype}")
    print("=" * 60)

    # 1. 配置量化参数
    if args.bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # NF4 量化
            bnb_4bit_compute_dtype=get_compute_dtype(args.compute_dtype),
            bnb_4bit_use_double_quant=args.double_quant,  # 双重量化
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    print("\n[1/5] 加载模型（量化配置）...")

    # 2. 检测模型类型
    model_type = detect_model_type(args.model)
    print(f"  检测到模型类型: {model_type}")

    # 3. 加载模型（根据类型选择正确的加载方式）
    # 注意：不要调用 prepare_model_for_kbit_training()，它会添加梯度钩子导致 save_pretrained 卡死
    if model_type == 'qwen3_vl':
        print("  使用 Qwen3 VL 多模态模型加载方式...")
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=get_compute_dtype(args.compute_dtype),
        )
        print("  ✓ 使用 Qwen3VLForConditionalGeneration 加载")

    elif model_type == 'qwen2_vl':
        print("  使用 Qwen2 VL 多模态模型加载方式...")
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=get_compute_dtype(args.compute_dtype),
        )
        print("  ✓ 使用 Qwen2VLForConditionalGeneration 加载")

    elif model_type == 'multimodal':
        print("  使用通用多模态模型加载方式...")
        model = AutoModel.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=get_compute_dtype(args.compute_dtype),
        )
        print("  ✓ 使用 AutoModel 加载")

    else:
        # 纯文本模型（qwen3_5_text, qwen_text, text_only）
        print("  使用纯文本模型加载方式...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=get_compute_dtype(args.compute_dtype),
        )
        print("  ✓ 使用 AutoModelForCausalLM 加载")

    # 4. 加载 tokenizer 和 processor
    print("\n[2/5] 加载 Tokenizer 和 Processor...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
    )

    # Qwen3-VL 多模态模型有 processor（处理图像和视频）
    try:
        processor = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=True,
        )
        has_processor = True
        print("  ✓ 检测到多模态 Processor")
    except Exception:
        has_processor = False
        processor = None
        print("  ⚠ 未检测到 Processor（非多模态模型）")

    # 5. 分析模型结构（不调用 prepare_model_for_kbit_training，避免保存卡死）
    print("\n[3/5] 分析模型结构...")

    # 打印直接子模块
    print("  直接子模块:")
    for name, module in model.named_children():
        print(f"    - {name}: {type(module).__name__}")

    # 使用 named_modules() 递归检测视觉编码器
    # Qwen3-VL 结构: model.visual 是视觉编码器，但 named_children() 只看一层
    has_visual = False
    visual_modules = []
    for name, module in model.named_modules():
        if 'visual' in name.lower() and name:  # name 非空且包含 visual
            has_visual = True
            visual_modules.append(f"{name} ({type(module).__name__})")

    if has_visual:
        print(f"\n  ✓ 检测到视觉编码器（多模态模型）:")
        for vm in visual_modules[:5]:  # 最多显示 5 个
            print(f"      {vm}")
        if len(visual_modules) > 5:
            print(f"      ... 共 {len(visual_modules)} 个视觉模块")
    else:
        print("\n  ⚠ 未检测到视觉编码器（可能是纯文本模型）")

    # 打印量化信息
    print("\n  量化配置:")
    print(f"    量化类型: {'NF4' if args.bits == 4 else 'INT8'}")
    print(f"    计算精度: {args.compute_dtype}")
    if args.bits == 4 and args.double_quant:
        print("    双重量化: 已启用（约节省 0.4 bit/参数）")

    # 6. 保存模型
    print(f"\n[4/5] 保存模型到 {args.output}...")
    print(f"  使用格式: {'safetensors' if args.use_safetensors else 'pytorch'}")
    print(f"  分片大小: {args.max_shard_size}")
    os.makedirs(args.output, exist_ok=True)

    # 保存模型权重（BnB NF4 量化权重，体积约为 FP16 的 1/4）
    print("  正在保存量化权重（可能需要几分钟，请耐心等待）...")
    try:
        model.save_pretrained(
            args.output,
            max_shard_size=args.max_shard_size,
            safe_serialization=args.use_safetensors,
        )
        print("  ✓ 模型权重保存完成")
    except Exception as e:
        print(f"  ⚠ safetensors 保存失败: {e}")
        print("  尝试使用 pytorch 格式保存...")
        model.save_pretrained(
            args.output,
            max_shard_size=args.max_shard_size,
            safe_serialization=False,
        )
        print("  ✓ 模型权重保存完成（pytorch 格式）")

    # 保存 tokenizer
    tokenizer.save_pretrained(args.output)
    print("  ✓ Tokenizer 保存完成")

    # 保存 processor（多模态模型）
    if has_processor and processor is not None:
        processor.save_pretrained(args.output)
        print("  ✓ Processor 保存完成")

    # 7. 修补配置，确保 vLLM 兼容
    print(f"\n[5/5] 修补配置（vLLM 兼容）...")
    quant_type = "nf4" if args.bits == 4 else "int8"
    patch_config_for_vllm(args.output, args.bits, quant_type)

    # 可选：从官方模型复制配置文件
    if args.copy_config and args.official_model:
        print("  从官方模型复制配置文件...")
        copy_config_files(args.official_model, args.output)
        # 重新修补（因为复制会覆盖）
        patch_config_for_vllm(args.output, args.bits, quant_type)

    # 显示保存的文件列表
    print("\n  保存的文件:")
    total_size = 0
    for f in sorted(os.listdir(args.output)):
        fpath = os.path.join(args.output, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            total_size += size
            if size > 1024 * 1024:
                print(f"    {f}: {size / (1024 * 1024):.2f} MB")
    print(f"  总大小: {total_size / (1024 * 1024):.2f} MB")

    # 推送到 Hub
    if args.push_to_hub:
        print(f"\n  推送到 {args.push_to_hub}...")
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        if has_processor and processor is not None:
            processor.push_to_hub(args.push_to_hub)

    print("\n" + "=" * 60)
    print("✓ 量化完成!")
    print("=" * 60)

    # 统计信息
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count / 1e9:.2f}B")
    print(f"输出目录: {args.output}")

    # 推荐下一步
    print("\n推荐下一步:")
    print("  1. 验证量化模型:")
    print(f"     python verify_bnb.py --model {args.output} --bits {args.bits}")
    print("  2. 使用 vLLM 部署:")
    print(f"     vllm serve {args.output} \\")
    print(f"         --quantization bitsandbytes \\")
    print(f"         --load-format bitsandbytes \\")
    print(f"         --dtype bfloat16 \\")
    print(f"         --max-model-len 8192 \\")
    print(f"         --port 7890")
    print("  3. 使用 ms-swift 微调:")
    print(f"     swift sft --model {args.output} --train_type lora ...")


if __name__ == "__main__":
    main()
