"""
GPTQ 量化脚本 - 使用 GPTQModel 进行 GPTQ 量化（支持 Qwen3.5 等新架构）

用法：
    python quantize_gptq.py \
        --model /path/to/merged_model \
        --output /path/to/gptq_model \
        --dataset /path/to/calibration.jsonl \
        --bits 4 \
        --gpu 0

依赖安装：
    pip install gptqmodel optimum accelerate

说明：
    GPTQModel 原生支持 Qwen3.5（包括 dense/MoE/多模态/纯文本），无需额外补丁。
    校准数据支持 JSONL 格式，自动识别 text/query/response/instruction 等字段。
"""
import os
import json
import shutil
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='GPTQ 量化脚本（使用 GPTQModel）')
    parser.add_argument('--model', type=str, required=True, help='合并后的模型路径')
    parser.add_argument('--output', type=str, required=True, help='量化后模型保存路径')
    parser.add_argument('--bits', type=int, default=4, choices=[2, 3, 4, 5, 6, 8], help='量化位数（默认 4-bit）')
    parser.add_argument('--group_size', type=int, default=128, help='量化分组大小（默认 128）')
    parser.add_argument('--dataset', type=str, default=None, help='校准数据集 JSONL 路径')
    parser.add_argument('--max_calib_samples', type=int, default=128, help='校准样本数量')
    parser.add_argument('--max_calib_seq_len', type=int, default=512, help='校准最大序列长度')
    parser.add_argument('--sym', action='store_true', default=True, help='对称量化（默认开启）')
    parser.add_argument('--desc_act', action='store_true', default=True, help='激活排序（默认开启，提升精度）')
    parser.add_argument('--true_sequential', action='store_true', default=True, help='真正顺序量化（默认开启）')
    parser.add_argument('--damp_percent', type=float, default=0.01, help='Hessian 阻尼系数（默认 0.01）')
    parser.add_argument('--batch_size', type=int, default=1, help='量化批次大小（默认 1）')
    parser.add_argument('--lm_head', action='store_true', default=False, help='是否量化 lm_head（默认不量化）')
    parser.add_argument('--copy_config', action='store_true', help='复制官方模型配置文件')
    parser.add_argument('--official_model', type=str, default=None, help='官方模型路径（用于复制配置）')
    parser.add_argument('--gpu', type=str, default=None, help='指定使用的 GPU，如 "0" 或 "0,1"')
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


def load_calibration_data(dataset_path, max_samples):
    """加载校准数据集，返回文本字符串列表

    支持 JSONL 格式，自动识别以下字段组合：
    - text（直接使用）
    - query + response
    - instruction + input + output
    - prompt + completion
    """
    if dataset_path is None:
        return None

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f'数据集文件不存在: {dataset_path}')
    if dataset_path.is_dir():
        raise ValueError(f'数据集路径是一个目录，需要指定 JSONL 文件: {dataset_path}')

    texts = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 按优先级识别字段
            if 'text' in item:
                texts.append(item['text'])
            elif 'query' in item or 'response' in item:
                query = item.get('query', '')
                response = item.get('response', '')
                texts.append(f'{query} {response}'.strip())
            elif 'instruction' in item:
                parts = [item.get('instruction', ''), item.get('input', ''), item.get('output', '')]
                texts.append(' '.join(p for p in parts if p).strip())
            elif 'prompt' in item or 'completion' in item:
                prompt = item.get('prompt', '')
                completion = item.get('completion', '')
                texts.append(f'{prompt} {completion}'.strip())

            if len(texts) >= max_samples:
                break

    if not texts:
        raise ValueError(f'数据集中未提取到有效文本，请检查字段名（支持: text/query/response/instruction/prompt/completion）')

    # 随机打乱并截取
    import random
    random.seed(42)
    random.shuffle(texts)
    texts = texts[:max_samples]

    print(f'加载本地数据集: {dataset_path} ({len(texts)} 条)')
    return texts


def quantize_model(args):
    """执行 GPTQ 量化"""
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print(f'使用 GPU: {args.gpu}')

    from gptqmodel import GPTQModel, GPTQConfig
    from transformers import AutoTokenizer

    print('=' * 60)
    print('GPTQ 量化脚本（GPTQModel）')
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
    print(f'对称量化: {args.sym}')
    print(f'激活排序: {args.desc_act}')
    print(f'顺序量化: {args.true_sequential}')
    print(f'阻尼系数: {args.damp_percent}')
    print(f'量化 lm_head: {args.lm_head}')

    # 步骤 1: 创建量化配置
    print('\n步骤 1: 创建量化配置...')
    quant_config = GPTQConfig(
        bits=args.bits,
        group_size=args.group_size,
        sym=args.sym,
        desc_act=args.desc_act,
        true_sequential=args.true_sequential,
        damp_percent=args.damp_percent,
        lm_head=args.lm_head,
    )

    # 步骤 2: 加载模型
    print('步骤 2: 加载模型...')
    model = GPTQModel.load(
        str(model_path),
        quantize_config=quant_config,
        trust_remote_code=True,
    )

    # 步骤 3: 加载校准数据集
    print('步骤 3: 加载校准数据集...')
    calib_data = load_calibration_data(args.dataset, args.max_calib_samples)
    if calib_data is None:
        print('未指定校准数据集，将使用模型默认校准方式')
        calib_data = []

    # 步骤 4: 执行量化
    print('步骤 4: 执行量化...')
    quantize_kwargs = {
        'batch_size': args.batch_size,
    }
    if calib_data:
        quantize_kwargs['calibration_data'] = calib_data

    model.quantize(**quantize_kwargs)

    # 步骤 5: 保存量化模型
    print('步骤 5: 保存量化模型...')
    model.save(str(output_path))

    # 保存分词器
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(str(output_path))

    # 复制配置文件
    if args.copy_config and args.official_model:
        print('\n步骤 6: 复制配置文件...')
        copy_config_files(args.official_model, output_path)

    # 保存量化信息
    quant_info = {
        'model_path': str(model_path),
        'output_path': str(output_path),
        'quant_method': 'gptq',
        'quant_bits': args.bits,
        'group_size': args.group_size,
        'sym': args.sym,
        'desc_act': args.desc_act,
        'true_sequential': args.true_sequential,
        'damp_percent': args.damp_percent,
        'lm_head': args.lm_head,
        'dataset': args.dataset or 'default',
        'max_calib_samples': args.max_calib_samples,
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
