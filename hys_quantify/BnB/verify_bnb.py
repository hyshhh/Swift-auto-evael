#!/usr/bin/env python3
"""
BnB 量化模型验证脚本 - 支持多模态
验证量化后的模型能否正常加载和推理
"""

import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from PIL import Image
import requests


def parse_args():
    parser = argparse.ArgumentParser(description="验证 BnB 量化模型")
    parser.add_argument("--model", type=str, required=True, help="量化模型路径")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="量化位数")
    parser.add_argument("--text", type=str, default="你好，请介绍一下你自己", help="测试文本")
    parser.add_argument("--image", type=str, default=None, help="测试图像路径（可选）")
    parser.add_argument("--max_tokens", type=int, default=256, help="最大生成长度")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("BnB 量化模型验证")
    print("=" * 60)
    print(f"模型路径: {args.model}")
    print(f"量化位数: {args.bits}-bit")
    print("=" * 60)

    # 1. 配置量化
    if args.bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    # 2. 加载模型
    print("\n[1/3] 加载量化模型...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        print("  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        return

    # 3. 加载 tokenizer
    print("[2/3] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
    )
    print("  ✓ Tokenizer 加载成功")

    # 检测是否为多模态模型
    has_processor = False
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=True,
        )
        has_processor = True
        print("  ✓ 检测到多模态 Processor")
    except Exception:
        print("  ⚠ 未检测到 Processor（纯文本模型）")

    # 4. 文本推理测试
    print("\n[3/3] 文本推理测试...")
    print(f"  输入: {args.text}")

    messages = [{"role": "user", "content": args.text}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=0.7,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  输出: {response}")

    # 5. 多模态测试（如果有图像）
    if has_processor and processor is not None and args.image:
        print("\n多模态推理测试...")
        try:
            if args.image.startswith("http"):
                image = Image.open(requests.get(args.image, stream=True).raw)
            else:
                image = Image.open(args.image)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "请描述这张图片"},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    temperature=0.7,
                )

            response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"  图像描述: {response}")
            print("  ✓ 多模态推理成功")
        except Exception as e:
            print(f"  ✗ 多模态推理失败: {e}")

    # 6. 统计信息
    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)

    # 显存占用
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"显存占用: {mem_used:.2f} GB (已分配) / {mem_reserved:.2f} GB (已保留)")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count / 1e9:.2f}B")

    print("\n量化模型验证通过! ✓")


if __name__ == "__main__":
    main()
