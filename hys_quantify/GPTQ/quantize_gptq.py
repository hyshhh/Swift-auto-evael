"""
GPTQ 量化脚本 - 使用 GPTQModel 进行 GPTQ 量化（支持 Qwen3-VL 多模态架构）

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
    GPTQModel 原生支持 Qwen3-VL 多模态模型，同时量化语言模型和视觉编码器。
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
        'processor_config.json',
        'chat_template.json',
        'special_tokens_map.json',
        'tokenizer.json',
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
    - messages（对话格式）

    对于多模态模型，返回包含图像路径的字典列表
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

            # 检查是否为多模态数据（包含图像）
            has_images = 'images' in item or 'image' in item

            # 按优先级识别字段
            if 'text' in item:
                if has_images:
                    texts.append({
                        'text': item['text'],
                        'images': item.get('images', item.get('image', []))
                    })
                else:
                    texts.append(item['text'])
            elif 'query' in item or 'response' in item:
                query = item.get('query', '')
                response = item.get('response', '')
                text = f'{query} {response}'.strip()
                if has_images:
                    texts.append({
                        'text': text,
                        'images': item.get('images', item.get('image', []))
                    })
                else:
                    texts.append(text)
            elif 'instruction' in item:
                parts = [item.get('instruction', ''), item.get('input', ''), item.get('output', '')]
                text = ' '.join(p for p in parts if p).strip()
                if has_images:
                    texts.append({
                        'text': text,
                        'images': item.get('images', item.get('image', []))
                    })
                else:
                    texts.append(text)
            elif 'prompt' in item or 'completion' in item:
                prompt = item.get('prompt', '')
                completion = item.get('completion', '')
                text = f'{prompt} {completion}'.strip()
                if has_images:
                    texts.append({
                        'text': text,
                        'images': item.get('images', item.get('image', []))
                    })
                else:
                    texts.append(text)
            elif 'messages' in item:
                # 对话格式，提取所有文本
                text_parts = []
                for msg in item['messages']:
                    if isinstance(msg.get('content'), str):
                        text_parts.append(msg['content'])
                    elif isinstance(msg.get('content'), list):
                        for content in msg['content']:
                            if isinstance(content, dict) and content.get('type') == 'text':
                                text_parts.append(content['text'])
                text = ' '.join(text_parts)
                if has_images:
                    texts.append({
                        'text': text,
                        'images': item.get('images', item.get('image', []))
                    })
                else:
                    texts.append(text)

            if len(texts) >= max_samples:
                break

    if not texts:
        raise ValueError(f'数据集中未提取到有效文本，请检查字段名（支持: text/query/response/instruction/prompt/completion/messages）')

    # 随机打乱并截取
    import random
    random.seed(42)
    random.shuffle(texts)
    texts = texts[:max_samples]

    print(f'加载本地数据集: {dataset_path} ({len(texts)} 条)')
    return texts


def detect_model_type(model_path):
    """检测模型类型，返回正确的加载类名"""
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        model_type = config.get('model_type', '')

        # 检查是否为 Qwen3 VL 多模态模型
        if model_type == 'qwen3_vl':
            return 'qwen3_vl'
        elif model_type == 'qwen2_vl':
            return 'qwen2_vl'
        elif 'visual_config' in config or 'vision_config' in config:
            return 'multimodal'
        elif model_type == 'qwen3_5':
            return 'qwen3_5_text'
        elif 'qwen' in model_type:
            return 'qwen_text'

    return 'text_only'


def quantize_model(args):
    """执行 GPTQ 量化"""
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print(f'使用 GPU: {args.gpu}')

    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer

    print('=' * 60)
    print('GPTQ 量化脚本（GPTQModel）')
    print('=' * 60)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f'模型路径不存在: {model_path}')

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # 检测模型类型
    model_type = detect_model_type(model_path)
    print(f'检测到模型类型: {model_type}')

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
    quant_config = QuantizeConfig(
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
    # 修补模型 config.json 中的 torch_dtype，避免 gptqmodel 的 auto_dtype 报错
    # （gptqmodel 内部自行加载 config，不接受 "auto" 字符串，必须是 torch.dtype）
    import torch
    config_json_path = Path(model_path) / 'config.json'
    config_modified = False
    with open(config_json_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    if config_dict.get('torch_dtype') == 'auto':
        config_dict['torch_dtype'] = 'bfloat16'
        with open(config_json_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        config_modified = True
        print('  ⚠ config.json 中 torch_dtype 为 "auto"，已临时修改为 "bfloat16"')
    model = GPTQModel.from_pretrained(
        str(model_path),
        quantize_config=quant_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # 恢复原始 config.json
    if config_modified:
        config_dict['torch_dtype'] = 'auto'
        with open(config_json_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print('  ✓ 已恢复 config.json 中的 torch_dtype 为 "auto"')

    # 打印模型结构
    print('\n模型结构:')
    for name, module in model.named_children():
        print(f'  - {name}: {type(module).__name__}')

    # 检查是否包含视觉编码器
    has_visual = hasattr(model, 'visual') or any('visual' in name for name, _ in model.named_children())
    if has_visual:
        print('  ✓ 包含视觉编码器（多模态模型）')
    else:
        print('  ⚠ 未检测到视觉编码器（可能是纯文本模型）')

    # 步骤 3: 加载校准数据集
    print('步骤 3: 加载校准数据集...')
    calib_data = load_calibration_data(args.dataset, args.max_calib_samples)
    if calib_data is None:
        print('未指定校准数据集，将使用模型默认校准方式')
        calib_data = []

    # 步骤 4: 执行量化
    print('步骤 4: 执行量化...')

    # 提取纯文本列表（新版 gptqmodel 只接受文本列表）
    if calib_data and isinstance(calib_data[0], dict):
        calib_texts = [item['text'] for item in calib_data if 'text' in item]
    else:
        calib_texts = calib_data if calib_data else []

    # gptqmodel 量化方式（calibration_dataset 作为位置参数，batch_size 作为关键字参数）
    if calib_texts:
        model.quantize(calibration_dataset=calib_texts, batch_size=args.batch_size)
    else:
        model.quantize()

    # 步骤 5: 保存量化模型
    print('步骤 5: 保存量化模型...')
    try:
        model.save(str(output_path))
    except AttributeError:
        model.save_quantized(str(output_path))

    # 保存分词器
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(str(output_path))

    # 保存 processor（多模态模型）
    if has_visual:
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
            processor.save_pretrained(str(output_path))
            print('✓ Processor 保存完成')
        except Exception as e:
            print(f'⚠ Processor 保存失败: {e}')

    # 自动从原模型复制配置文件（无论是否指定 --copy_config）
    print('\n步骤 6: 复制配置文件...')
    copy_config_files(model_path, output_path)

    # 如果指定了官方模型，也从官方模型复制（覆盖）
    if args.copy_config and args.official_model and args.official_model != str(model_path):
        print('从官方模型复制配置文件...')
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
        'model_type': model_type,
        'has_visual': has_visual,
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
