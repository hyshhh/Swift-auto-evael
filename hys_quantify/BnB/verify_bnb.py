#!/usr/bin/env python3
"""
BnB 量化模型验证脚本 - 支持 Qwen3-VL 多模态
验证量化后的模型能否正常加载和推理

用法:
    python verify_bnb.py --model /path/to/bnb-model --bits 4
    python verify_bnb.py --model /path/to/bnb-model --bits 4 --image /path/to/image.jpg
"""

import os
import json
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


def detect_model_type(model_path):
    """检测模型类型"""
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        model_type = config.get('model_type', '')
        if model_type == 'qwen3_vl':
            return 'qwen3_vl'
        elif model_type == 'qwen2_vl':
            return 'qwen2_vl'
        elif 'visual_config' in config or 'vision_config' in config:
            return 'multimodal'
    return 'text_only'


def parse_args():
    parser = argparse.ArgumentParser(description="验证 BnB 量化模型")
    parser.add_argument("--model", type=str, required=True, help="量化模型路径")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="量化位数")
    parser.add_argument("--text", type=str, default="你好，请介绍一下你自己", help="测试文本")
    parser.add_argument("--image", type=str, default=None, help="测试图像路径或URL（可选）")
    parser.add_argument("--video", type=str, default=None, help="测试视频路径或URL（可选）")
    parser.add_argument("--max_tokens", type=int, default=256, help="最大生成长度")
    return parser.parse_args()


def load_model(model_path, bits):
    """加载量化模型（自动检测多模态）"""
    # 配置量化
    if bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model_type = detect_model_type(model_path)
    print(f"  检测到模型类型: {model_type}")

    # 根据类型选择加载方式
    if model_type == 'qwen3_vl':
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        print("  ✓ 使用 Qwen3VLForConditionalGeneration 加载")
    elif model_type == 'qwen2_vl':
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        print("  ✓ 使用 Qwen2VLForConditionalGeneration 加载")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        print("  ✓ 使用 AutoModelForCausalLM 加载")

    return model, model_type


def main():
    args = parse_args()

    print("=" * 60)
    print("BnB 量化模型验证")
    print("=" * 60)
    print(f"模型路径: {args.model}")
    print(f"量化位数: {args.bits}-bit")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/4] 加载量化模型...")
    try:
        model, model_type = load_model(args.model, args.bits)
        print("  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        return

    # 2. 加载 tokenizer
    print("\n[2/4] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
    )
    print("  ✓ Tokenizer 加载成功")

    # 加载 processor（多模态模型）
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

    # 3. 文本推理测试
    print("\n[3/4] 文本推理测试...")
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
    print("  ✓ 文本推理成功")

    # 4. 多模态测试（如果有图像或视频）
    if has_processor and processor is not None and (args.image or args.video):
        print("\n[4/4] 多模态推理测试...")
        try:
            from qwen_vl_utils import process_vision_info

            content = []
            if args.image:
                if args.image.startswith("http"):
                    content.append({"type": "image", "image": args.image})
                else:
                    content.append({"type": "image", "image": f"file://{args.image}"})
                print(f"  图像: {args.image}")

            if args.video:
                if args.video.startswith("http"):
                    content.append({"type": "video", "video": args.video, "max_pixels": 128*32*32, "max_frames": 16})
                else:
                    content.append({"type": "video", "video": f"file://{args.video}", "max_pixels": 128*32*32, "max_frames": 16})
                print(f"  视频: {args.video}")

            content.append({"type": "text", "text": "请描述这张图片" if args.image else "请描述这个视频"})

            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                [messages], return_video_kwargs=True, image_patch_size=16, return_video_metadata=True
            )

            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
            else:
                video_metadatas = None

            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                video_metadata=video_metadatas, **video_kwargs,
                do_resize=False, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, temperature=0.7)

            response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"  输出: {response}")
            print("  ✓ 多模态推理成功")

        except Exception as e:
            print(f"  ✗ 多模态推理失败: {e}")
    else:
        print("\n[4/4] 跳过多模态测试（未提供图像/视频或非多模态模型）")

    # 5. 统计信息
    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)

    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"显存占用: {mem_used:.2f} GB (已分配) / {mem_reserved:.2f} GB (已保留)")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count / 1e9:.2f}B")

    # vLLM 部署提示
    print("\nvLLM 部署命令:")
    print(f"  vllm serve {args.model} \\")
    print(f"      --quantization bitsandbytes \\")
    print(f"      --load-format bitsandbytes \\")
    print(f"      --dtype bfloat16 \\")
    print(f"      --max-model-len 8192 \\")
    print(f"      --port 7890")

    print("\n量化模型验证通过! ✓")


if __name__ == "__main__":
    main()
