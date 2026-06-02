#!/usr/bin/env python3
"""
BnB NF4 量化脚本 - 适配 Qwen3.5 多模态模型
量化语言模型部分，视觉编码器保持 FP16/BF16

支持流式保存，避免卡住问题
"""

import os
import gc
import json
import argparse
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training


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


def parse_args():
    parser = argparse.ArgumentParser(description="BnB NF4 量化 Qwen3.5 多模态模型")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--output", type=str, required=True, help="输出路径")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="量化位数 (4=NF4, 8=INT8)")
    parser.add_argument("--double_quant", action="store_true", help="启用双重量化（进一步压缩）")
    parser.add_argument("--compute_dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"], help="计算精度")
    parser.add_argument("--save_merged", action="store_true", help="保存合并后的模型（不含量化配置）")
    parser.add_argument("--push_to_hub", type=str, default=None, help="推送到 HuggingFace Hub")
    parser.add_argument("--max_shard_size", type=str, default="5GB", help="每个分片最大大小")
    parser.add_argument("--use_safetensors", action="store_true", default=True, help="使用 safetensors 格式（推荐）")
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
    print("BnB NF4 量化 - Qwen3.5 多模态模型")
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

    print("\n[1/4] 加载模型（量化配置）...")

    # 2. 检测模型类型
    model_type = detect_model_type(args.model)
    print(f"  检测到模型类型: {model_type}")

    # 3. 加载模型（根据类型选择正确的加载方式）
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

    # 3. 加载 tokenizer 和 processor
    print("[2/4] 加载 Tokenizer 和 Processor...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
    )

    # Qwen3.5 多模态模型有 processor（处理图像）
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

    # 4. 准备模型用于训练（如果需要 QLoRA 微调）
    print("[3/4] 准备模型...")
    if args.bits == 4:
        model = prepare_model_for_kbit_training(model)

    # 打印模型信息
    print("\n模型结构:")
    for name, module in model.named_children():
        print(f"  - {name}: {type(module).__name__}")

    # 检查是否包含视觉编码器
    has_visual = hasattr(model, 'visual') or any('visual' in name for name, _ in model.named_children())
    if has_visual:
        print("  ✓ 包含视觉编码器（多模态模型）")
    else:
        print("  ⚠ 未检测到视觉编码器（可能是纯文本模型）")

    # 打印量化信息
    print("\n量化配置:")
    print(f"  量化类型: {'NF4' if args.bits == 4 else 'INT8'}")
    print(f"  计算精度: {args.compute_dtype}")
    if args.bits == 4 and args.double_quant:
        print("  双重量化: 已启用（约节省 0.4 bit/参数）")

    # 5. 保存模型（流式写入，避免卡住）
    print(f"\n[4/4] 保存模型到 {args.output}...")
    print(f"  使用格式: {'safetensors' if args.use_safetensors else 'pytorch'}")
    print(f"  分片大小: {args.max_shard_size}")
    os.makedirs(args.output, exist_ok=True)

    # 使用流式保存，避免内存占用过高
    print("  正在保存模型分片...")

    # 保存配置和分片权重
    model.save_pretrained(
        args.output,
        max_shard_size=args.max_shard_size,
        safe_serialization=args.use_safetensors,
    )
    print("  ✓ 模型权重保存完成")

    # 保存 tokenizer
    tokenizer.save_pretrained(args.output)
    print("  ✓ Tokenizer 保存完成")

    # 保存 processor（多模态模型）
    if has_processor and processor is not None:
        processor.save_pretrained(args.output)
        print("  ✓ Processor 保存完成")

    # 如果需要保存合并模型（不含量化配置，用于部署）
    if args.save_merged:
        print("\n  保存合并模型（反量化为 FP16）...")
        merged_path = os.path.join(args.output, "merged")
        os.makedirs(merged_path, exist_ok=True)

        # 释放当前模型内存
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # 重新加载原始模型（FP16）
        model_unmerged = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=get_compute_dtype(args.compute_dtype),
            low_cpu_mem_usage=True,  # 降低内存占用
        )

        # 流式保存合并模型
        model_unmerged.save_pretrained(
            merged_path,
            max_shard_size=args.max_shard_size,
            safe_serialization=args.use_safetensors,
        )
        tokenizer.save_pretrained(merged_path)
        if has_processor and processor is not None:
            processor.save_pretrained(merged_path)

        del model_unmerged
        gc.collect()
        torch.cuda.empty_cache()
        print("  ✓ 合并模型保存完成")

    # 推送到 Hub
    if args.push_to_hub:
        print(f"  推送到 {args.push_to_hub}...")
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
    print(f"     python verify_bnb.py --model {args.output}")
    print("  2. 使用 ms-swift 微调:")
    print(f"     swift sft --model {args.output} ...")
    print("  3. 使用 vLLM 部署（需要合并模型）:")


if __name__ == "__main__":
    main()
