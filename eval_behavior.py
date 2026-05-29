"""
行为识别模型评测脚本（纯 transformers + peft，不依赖 swift）
用法：
    CUDA_VISIBLE_DEVICES=2 python eval_behavior.py \
        --model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
        --adapters /media/ddc/新加卷/hys/qmy/ms-swift/output/v2-20260528-211924/checkpoint-31 \
        --dataset /media/ddc/新加卷/hys/qmy/agent/data/sft_val.jsonl
"""
import json
import re
import argparse
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description='行为识别评测')
    parser.add_argument('--model', type=str, required=True, help='基座模型路径')
    parser.add_argument('--adapters', type=str, default=None, help='LoRA adapter 路径')
    parser.add_argument('--dataset', type=str, required=True, help='评测数据集 jsonl 路径')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='最大生成 token 数')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    parser.add_argument('--debug', action='store_true', help='打印前几条的原始输出')
    parser.add_argument('--output', type=str, default=None, help='预测结果保存路径')
    return parser.parse_args()


def load_dataset(jsonl_path):
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def extract_behavior_id(text):
    """从模型输出中提取 behavior_id"""
    try:
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            bid = str(data.get('behavior_id', 'unknown'))
            return bid, data
    except (json.JSONDecodeError, KeyError):
        pass

    match = re.search(r'"behavior_id"\s*:\s*"?(\d+|unknown)"?', text)
    if match:
        return match.group(1), {}
    return 'unknown', {}


def load_model(model_path, adapter_path):
    """加载模型和处理器"""
    print(f'加载模型: {model_path}')

    # 尝试加载 VL 模型
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )
    except Exception:
        # 如果不是 Qwen2.5-VL，尝试 AutoModel
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    if adapter_path:
        print(f'加载 LoRA: {adapter_path}')
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, processor


def inference(model, processor, query, images, max_new_tokens=256):
    """单条推理"""
    # 构建消息
    content = [{'type': 'text', 'text': query}]
    for img_path in images:
        content.insert(0, {'type': 'image', 'image': img_path})

    messages = [{'role': 'user', 'content': content}]

    # 处理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=images if images else None, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 推理
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0,
            do_sample=False,
        )

    # 解码输出（只取新生成的部分）
    generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return output_text


def main():
    args = parse_args()

    # 加载数据
    samples = load_dataset(args.dataset)
    print(f'加载数据集: {args.dataset}，共 {len(samples)} 条样本')

    # GPU 信息
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')

    # 加载模型
    model, processor = load_model(args.model, args.adapters)

    # 评测统计
    stats = {
        'total': 0,
        'correct': 0,
        'unknown_pred': 0,
        'per_class': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'errors': [],
    }

    # 逐条推理
    for idx, sample in enumerate(tqdm(samples, desc='评测中')):
        query = sample['query']
        gt_response = sample['response']
        images = sample.get('images', [])

        # 解析 ground truth 的 behavior_id
        gt_id, _ = extract_behavior_id(gt_response)
        if gt_id == 'unknown':
            continue

        # 推理
        try:
            pred_text = inference(model, processor, query, images, args.max_new_tokens)
        except Exception as e:
            stats['errors'].append({'sample_idx': idx, 'error': str(e)})
            continue

        # 调试：打印原始输出
        if args.debug and stats['total'] < 3:
            print(f'\n--- 样本 {stats["total"]+1} ---')
            print(f'GT ID: {gt_id}')
            print(f'图片: {images[0] if images else "N/A"}')
            print(f'模型输出: {pred_text[:500]}')

        # 解析预测结果
        pred_id, _ = extract_behavior_id(pred_text)
        stats['total'] += 1
        stats['per_class'][gt_id]['total'] += 1

        if pred_id == 'unknown':
            stats['unknown_pred'] += 1

        if pred_id == gt_id:
            stats['correct'] += 1
            stats['per_class'][gt_id]['correct'] += 1
        else:
            stats['errors'].append({
                'gt_id': gt_id,
                'pred_id': pred_id,
                'pred_text': pred_text[:200],
                'image': images[0] if images else None,
            })

    # 输出结果
    print('\n' + '=' * 60)
    print('评测结果')
    print('=' * 60)

    total = stats['total']
    if total == 0:
        print('无有效样本')
        return

    accuracy = stats['correct'] / total
    print(f'总样本数: {len(samples)}')
    print(f'有效样本数: {total}')
    print(f'正确数: {stats["correct"]}')
    print(f'准确率: {accuracy:.4f} ({stats["correct"]}/{total})')
    print(f'未识别数: {stats["unknown_pred"]}')
    print(f'推理错误数: {len(stats["errors"])}')

    print(f'\n{"类别":<15} {"正确":<8} {"总数":<8} {"准确率":<10}')
    print('-' * 45)
    for cls_id in sorted(stats['per_class'].keys()):
        cls = stats['per_class'][cls_id]
        cls_acc = cls['correct'] / cls['total'] if cls['total'] > 0 else 0
        print(f'{cls_id:<15} {cls["correct"]:<8} {cls["total"]:<8} {cls_acc:.4f}')

    # 保存预测错误样本
    if stats['errors']:
        print(f'\n前 5 个错误样本:')
        for i, err in enumerate(stats['errors'][:5]):
            print(f'  [{i+1}] GT: {err.get("gt_id", "?")} | Pred: {err.get("pred_id", "?")} | Image: {err.get("image", "N/A")}')

    # 保存结果
    output_path = args.output or str(Path(args.dataset).with_suffix('.eval_result.json'))
    result = {
        'accuracy': accuracy,
        'total': total,
        'correct': stats['correct'],
        'unknown_pred': stats['unknown_pred'],
        'per_class': {k: dict(v) for k, v in stats['per_class'].items()},
        'error_samples': stats['errors'][:20],
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n结果已保存到: {output_path}')


if __name__ == '__main__':
    main()
