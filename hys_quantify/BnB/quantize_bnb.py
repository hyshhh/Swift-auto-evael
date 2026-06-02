#!/usr/bin/env python3
"""
BnB NF4 量化脚本 - 适配 Qwen3.5 多模态模型
仅量化语言模型部分，视觉编码器保持 FP16/BF16
"""

import os
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training


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

    # 2. 加载模型
    # Qwen3.5 多模态模型需要 trust_remote_code
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=get_compute_dtype(args.compute_dtype),
    )

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

    # 打印量化信息
    print("\n量化配置:")
    print(f"  量化类型: {'NF4' if args.bits == 4 else 'INT8'}")
    print(f"  计算精度: {args.compute_dtype}")
    if args.bits == 4 and args.double_quant:
        print("  双重量化: 已启用（约节省 0.4 bit/参数）")

    # 5. 保存模型
    print(f"\n[4/4] 保存模型到 {args.output}...")
    os.makedirs(args.output, exist_ok=True)

    # 保存量化模型
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    if has_processor and processor is not None:
        processor.save_pretrained(args.output)

    # 如果需要保存合并模型（不含量化配置，用于部署）
    if args.save_merged:
        print("  保存合并模型...")
        merged_path = os.path.join(args.output, "merged")
        os.makedirs(merged_path, exist_ok=True)

        # 注意：合并后模型会变大（反量化为 FP16）
        from peft import PeftModel
        model_unmerged = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=get_compute_dtype(args.compute_dtype),
        )
        model_unmerged.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        if has_processor and processor is not None:
            processor.save_pretrained(merged_path)
        del model_unmerged

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
