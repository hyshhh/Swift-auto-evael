"""
AWQ 量化脚本 - 使用 llm-compressor 进行 AWQ 量化

用法：
    python quantize_awq.py \
        --model /path/to/merged_model \
        --output /path/to/awq_model \
        --dataset alpaca-en \
        --bits 4 \
        --gpu 0,1
"""
import os
import json
import shutil
import argparse
from pathlib import Path

# 禁用 compressed_tensors 的设备分发，避免 PyTorch 版本兼容性问题
os.environ['CT_DISABLE_DEVICE_DISPATCH'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='AWQ 量化脚本（使用 llm-compressor）')
    parser.add_argument('--model', type=str, required=True, help='合并后的模型路径')
    parser.add_argument('--output', type=str, required=True, help='量化后模型保存路径')
    parser.add_argument('--bits', type=int, default=4, choices=[2, 3, 4], help='量化位数（默认 4-bit）')
    parser.add_argument('--group_size', type=int, default=128, help='量化分组大小')
    parser.add_argument('--dataset', type=str, default='alpaca', help='校准数据集名称或路径')
    parser.add_argument('--copy_config', action='store_true', help='复制官方模型配置文件')
    parser.add_argument('--official_model', type=str, default=None, help='官方模型路径（用于复制配置）')
    parser.add_argument('--max_seq_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--num_calibration_samples', type=int, default=128, help='校准样本数量')
    parser.add_argument('--gpu', type=str, default=None, help='指定使用的 GPU，如 "0" 或 "0,1" 或 "1,2,3"')
    parser.add_argument('--max_shard_size', type=str, default='2GB', help='保存时每个分片的最大大小（默认 2GB）')
    return parser.parse_args()


def copy_config_files(official_model, output_path):
    """从官方模型复制配置文件"""
    official_model = Path(official_model)
    output_path = Path(output_path)

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

    for extra_file in ['processor_config.json', 'args.json']:
        extra_path = output_path / extra_file
        if extra_path.exists():
            extra_path.unlink()
            print(f'删除: {extra_file}')


def quantize_model(args):
    """执行 AWQ 量化"""
    # 在导入 torch 之前设置 GPU，确保只使用指定的显卡
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print(f'使用 GPU: {args.gpu}')

    import swift  # 注册 qwen3_5 等自定义模型类型到 transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier

    print('=' * 60)
    print('AWQ 量化脚本（llm-compressor）')
    print('=' * 60)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f'模型路径不存在: {model_path}')

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f'输入模型: {model_path}')
    print(f'输出路径: {output_path}')
    print(f'量化位数: {args.bits}-bit')
    print(f'分组大小: {args.group_size}')
    print(f'校准数据集: {args.dataset}')

    # 加载模型和分词器
    print('\n步骤 1: 加载模型...')
    try:
        from transformers import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype="auto",
            trust_remote_code=True,
        )
    except ImportError:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype="auto",
            trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    # 配置 AWQ + 量化修饰器
    print('步骤 2: 配置 AWQ 量化...')
    awq_modifier = AWQModifier(
        ignore=["lm_head"],
        scheme=f"W{args.bits}A16",
        targets=["Linear"],
        duo_scaling="both",
    )
    recipe = [awq_modifier]

    # 加载校准数据集
    print('步骤 3: 加载校准数据集...')
    dataset_path = Path(args.dataset)
    if dataset_path.exists() and dataset_path.is_file():
        from datasets import load_dataset
        ds = load_dataset("json", data_files=str(dataset_path), split="train")
        ds = ds.shuffle(seed=42).select(range(min(args.num_calibration_samples, len(ds))))

        def preprocess(example):
            text = example.get("query", "") + " " + example.get("response", "")
            return {"text": text}

        ds = ds.map(preprocess, remove_columns=ds.column_names)
        dataset = ds
        print(f'加载本地数据集: {dataset_path} ({len(ds)} 条)')
    elif dataset_path.exists() and dataset_path.is_dir():
        raise ValueError(f'数据集路径是一个目录，需要指定 JSONL 文件: {dataset_path}')
    else:
        # 检查是否是注册的数据集名称
        registered_datasets = ['c4', 'cnn-dailymail', 'custom', 'evolcodealpaca',
                              'flickr', 'gsm8k', 'open-platypus', 'peoples-speech',
                              'ultrachat-200k', 'wikitext']
        if args.dataset in registered_datasets:
            dataset = args.dataset
            print(f'使用注册数据集: {args.dataset}')
        else:
            raise FileNotFoundError(
                f'数据集文件不存在: {dataset_path}\n'
                f'请检查路径是否正确，或使用以下注册数据集之一: {registered_datasets}'
            )

    # 执行量化
    print('步骤 4: 执行量化...')

    # Monkey-patch: 绕过 torch.accelerator.get_memory_info 兼容性问题
    import compressed_tensors.offload.dispatch as dispatch_module
    def patched_get_device_memory():
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            return {
                torch.device("cuda", i): torch.cuda.get_device_properties(i).total_memory
                for i in range(device_count)
            }
        return {torch.device("cpu"): 0}
    dispatch_module.get_device_memory = patched_get_device_memory

    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
    )

    # 保存模型：先移到 CPU 避免 offloaded modules 导致 OOM
    print('步骤 5: 保存量化模型...')
    import torch
    model.to("cpu")
    torch.cuda.empty_cache()
    model.save_pretrained(str(output_path), save_compressed=False, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(str(output_path))

    # 复制配置文件
    if args.copy_config and args.official_model:
        print('\n步骤 6: 复制配置文件...')
        copy_config_files(args.official_model, output_path)

    # 保存量化信息
    quant_info = {
        'model_path': str(model_path),
        'output_path': str(output_path),
        'quant_method': 'awq',
        'quant_bits': args.bits,
        'group_size': args.group_size,
        'dataset': args.dataset,
    }
    info_path = output_path / 'quant_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(quant_info, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 60)
    print('量化完成！')
    print('=' * 60)
    print(f'量化模型保存到: {output_path}')

    total_size = 0
    for file in output_path.iterdir():
        if file.is_file():
            size = file.stat().st_size
            total_size += size
            if size > 1024 * 1024:
                print(f'  {file.name}: {size / 1024 / 1024:.2f} MB')

    print(f'\n总大小: {total_size / 1024 / 1024:.2f} MB')

    return output_path


def main():
    args = parse_args()
    quantize_model(args)


if __name__ == '__main__':
    main()
