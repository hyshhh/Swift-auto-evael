"""
AWQ 量化脚本 - 将 LoRA 微调后的模型量化为 AWQ 格式

用法：
    python quantize_awq.py \
        --model /path/to/merged_model \
        --output /path/to/awq_model \
        --dataset alpaca-en \
        --bits 4
"""
import os
import json
import shutil
import argparse
from pathlib import Path

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='AWQ 量化脚本')
    parser.add_argument('--model', type=str, required=True, help='合并后的模型路径')
    parser.add_argument('--output', type=str, required=True, help='量化后模型保存路径')
    parser.add_argument('--bits', type=int, default=4, choices=[4], help='量化位数（默认 4-bit）')
    parser.add_argument('--group_size', type=int, default=128, help='量化分组大小')
    parser.add_argument('--dataset', type=str, default=None, help='校准数据集路径（可选）')
    parser.add_argument('--copy_config', action='store_true', help='复制官方模型配置文件')
    parser.add_argument('--official_model', type=str, default=None, help='官方模型路径（用于复制配置）')
    return parser.parse_args()


def copy_config_files(official_model, output_path):
    """从官方模型复制配置文件"""
    official_model = Path(official_model)
    output_path = Path(output_path)

    # 需要复制的配置文件
    config_files = [
        'config.json',
        'video_preprocessor_config.json',
        'tokenizer_config.json',
        'vocab.json',
        'merges.txt',
        'preprocessor_config.json',
        'generation_config.json',
    ]

    for config_file in config_files:
        src = official_model / config_file
        dst = output_path / config_file
        if src.exists():
            shutil.copy2(src, dst)
            print(f'复制: {config_file}')
        else:
            print(f'跳过: {config_file}（不存在）')

    # 删除可能多余的文件
    for extra_file in ['processor_config.json', 'args.json']:
        extra_path = output_path / extra_file
        if extra_path.exists():
            extra_path.unlink()
            print(f'删除: {extra_file}')


def quantize_model(args):
    """执行 AWQ 量化"""
    print('=' * 60)
    print('AWQ 量化脚本')
    print('=' * 60)

    # 检查模型路径
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f'模型路径不存在: {model_path}')

    # 检查输出路径
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f'输入模型: {model_path}')
    print(f'输出路径: {output_path}')
    print(f'量化位数: {args.bits}-bit')
    print(f'分组大小: {args.group_size}')

    # 加载模型
    print('\n步骤 1: 加载模型...')
    model = AutoAWQForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
    )

    # 加载分词器
    print('步骤 2: 加载分词器...')
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=True,
    )

    # 量化配置
    quant_config = {
        'zero_point': True,
        'q_group_size': args.group_size,
        'w_bit': args.bits,
        'version': 'GEMM',
    }

    print(f'\n步骤 3: 开始量化...')
    print(f'量化配置: {quant_config}')

    # 执行量化
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        n_parallel_workers=4,
    )

    # 保存量化后的模型
    print(f'\n步骤 4: 保存量化模型...')
    model.save_quantized(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # 复制配置文件
    if args.copy_config and args.official_model:
        print('\n步骤 5: 复制配置文件...')
        copy_config_files(args.official_model, output_path)

    # 保存量化信息
    quant_info = {
        'model_path': str(model_path),
        'output_path': str(output_path),
        'quant_bits': args.bits,
        'group_size': args.group_size,
        'quant_config': quant_config,
    }
    info_path = output_path / 'quant_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(quant_info, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 60)
    print('量化完成！')
    print('=' * 60)
    print(f'量化模型保存到: {output_path}')

    # 显示文件大小
    total_size = 0
    for file in output_path.iterdir():
        if file.is_file():
            size = file.stat().st_size
            total_size += size
            if size > 1024 * 1024:  # 大于 1MB 显示
                print(f'  {file.name}: {size / 1024 / 1024:.2f} MB')

    print(f'\n总大小: {total_size / 1024 / 1024:.2f} MB')

    return output_path


def main():
    args = parse_args()
    quantize_model(args)


if __name__ == '__main__':
    main()
