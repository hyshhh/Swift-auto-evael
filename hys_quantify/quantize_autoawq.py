"""
AWQ 量化脚本 - 使用 AutoAWQ 进行 AWQ 量化（标准 AWQ 格式，vLLM 兼容）

用法：
    python quantize_autoawq.py \
        --model /path/to/merged_model \
        --output /path/to/awq_model \
        --dataset /path/to/calibration.jsonl \
        --bits 4 \
        --gpu 0,1
"""
import os
import json
import shutil
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='AWQ 量化脚本（使用 AutoAWQ，vLLM 兼容）')
    parser.add_argument('--model', type=str, required=True, help='合并后的模型路径')
    parser.add_argument('--output', type=str, required=True, help='量化后模型保存路径')
    parser.add_argument('--bits', type=int, default=4, choices=[3, 4], help='量化位数（默认 4-bit）')
    parser.add_argument('--group_size', type=int, default=128, help='量化分组大小')
    parser.add_argument('--dataset', type=str, default=None, help='校准数据集 JSONL 路径（默认使用 pileval）')
    parser.add_argument('--copy_config', action='store_true', help='复制官方模型配置文件')
    parser.add_argument('--official_model', type=str, default=None, help='官方模型路径（用于复制配置）')
    parser.add_argument('--max_calib_samples', type=int, default=128, help='校准样本数量')
    parser.add_argument('--max_calib_seq_len', type=int, default=512, help='校准最大序列长度')
    parser.add_argument('--n_parallel_calib_samples', type=int, default=None, help='并行校准样本数（降低显存占用）')
    parser.add_argument('--gpu', type=str, default=None, help='指定使用的 GPU，如 "0" 或 "0,1" 或 "1,2,3"')
    parser.add_argument('--zero_point', action='store_true', default=True, help='使用零点量化（默认开启）')
    parser.add_argument('--version', type=str, default='GEMM', choices=['GEMM', 'Marlin'], help='量化内核版本')
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
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print(f'使用 GPU: {args.gpu}')

    import swift  # 注册 qwen3_5 等自定义模型类型到 transformers

    # AutoAWQ 不认识 qwen3_5，monkey-patch 注册复用 Qwen3 处理器
    from awq.models.auto import AWQ_CAUSAL_LM_MODEL_MAP
    from awq.models.base import TRANSFORMERS_AUTO_MAPPING_DICT
    if "qwen3_5" not in AWQ_CAUSAL_LM_MODEL_MAP:
        from awq.models.qwen3 import Qwen3AWQForCausalLM
        AWQ_CAUSAL_LM_MODEL_MAP["qwen3_5"] = Qwen3AWQForCausalLM
        TRANSFORMERS_AUTO_MAPPING_DICT["qwen3_5"] = "AutoModelForCausalLM"
        print('已注册 qwen3_5 到 AutoAWQ（复用 Qwen3 处理器）')

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print('=' * 60)
    print('AWQ 量化脚本（AutoAWQ，vLLM 兼容格式）')
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
    print(f'量化内核: {args.version}')

    # 量化配置
    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.group_size,
        "w_bit": args.bits,
        "version": args.version,
    }

    # 加载模型和分词器
    print('\n步骤 1: 加载模型...')
    model = AutoAWQForCausalLM.from_pretrained(
        str(model_path),
        low_cpu_mem_usage=True,
        use_cache=False,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    # 加载校准数据集
    print('步骤 2: 加载校准数据集...')
    calib_data = None
    if args.dataset is not None:
        dataset_path = Path(args.dataset)
        if dataset_path.exists() and dataset_path.is_file():
            from datasets import load_dataset
            ds = load_dataset("json", data_files=str(dataset_path), split="train")
            ds = ds.shuffle(seed=42).select(range(min(args.max_calib_samples, len(ds))))

            def preprocess(example):
                return example.get("query", "") + " " + example.get("response", "")

            calib_data = [preprocess(example) for example in ds]
            print(f'加载本地数据集: {dataset_path} ({len(calib_data)} 条)')
        elif dataset_path.exists() and dataset_path.is_dir():
            raise ValueError(f'数据集路径是一个目录，需要指定 JSONL 文件: {dataset_path}')
        else:
            raise FileNotFoundError(f'数据集文件不存在: {dataset_path}')
    else:
        print('使用默认校准数据集: pileval')

    # 执行量化
    print('步骤 3: 执行量化...')
    quantize_kwargs = {
        "quant_config": quant_config,
        "max_calib_samples": args.max_calib_samples,
        "max_calib_seq_len": args.max_calib_seq_len,
    }
    if calib_data is not None:
        quantize_kwargs["calib_data"] = calib_data
    if args.n_parallel_calib_samples is not None:
        quantize_kwargs["n_parallel_calib_samples"] = args.n_parallel_calib_samples

    model.quantize(tokenizer, **quantize_kwargs)

    # 保存模型
    print('步骤 4: 保存量化模型...')
    model.save_quantized(str(output_path), safetensors=True)
    tokenizer.save_pretrained(str(output_path))

    # 复制配置文件
    if args.copy_config and args.official_model:
        print('\n步骤 5: 复制配置文件...')
        copy_config_files(args.official_model, output_path)

    # 保存量化信息
    quant_info = {
        'model_path': str(model_path),
        'output_path': str(output_path),
        'quant_method': 'awq',
        'quant_bits': args.bits,
        'group_size': args.group_size,
        'version': args.version,
        'zero_point': args.zero_point,
        'dataset': args.dataset or 'pileval',
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
