"""
行为识别模型评测脚本（vLLM API 模式）

用法：
    python eval_behavior.py \
        --vllm_url http://localhost:7890 \
        --model_name Qwen/Qwen3-VL-4B-AWQ \
        --dataset /media/ddc/新加卷/hys/qmy/agent/data/sft_val.jsonl \
        --debug
"""
import json
import re
import base64
import argparse
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='行为识别评测')
    parser.add_argument('--vllm_url', type=str, default='http://localhost:7890', help='vLLM 服务地址')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-4B-AWQ', help='vLLM 模型名称')
    parser.add_argument('--api_key', type=str, default='abc123', help='API 密钥')
    parser.add_argument('--dataset', type=str, required=True, help='评测数据集 jsonl 路径')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='最大生成 token 数')
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


def encode_image(image_path):
    """将图片编码为 base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def inference_vllm(args, query, images):
    """通过 vLLM API 进行推理"""
    content = []

    for img_path in images:
        if img_path.startswith('http'):
            content.append({
                'type': 'image_url',
                'image_url': {'url': img_path}
            })
        else:
            img_base64 = encode_image(img_path)
            content.append({
                'type': 'image_url',
                'image_url': {'url': f'data:image/jpeg;base64,{img_base64}'}
            })

    content.append({'type': 'text', 'text': query})

    messages = [{'role': 'user', 'content': content}]

    payload = {
        'model': args.model_name,
        'messages': messages,
        'max_tokens': args.max_new_tokens,
        'temperature': 0,
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {args.api_key}',
    }

    response = requests.post(
        f'{args.vllm_url}/v1/chat/completions',
        json=payload,
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()

    result = response.json()
    return result['choices'][0]['message']['content']


def main():
    args = parse_args()

    samples = load_dataset(args.dataset)
    print(f'加载数据集: {args.dataset}，共 {len(samples)} 条样本')
    print(f'vLLM 服务: {args.vllm_url}')
    print(f'模型名称: {args.model_name}')

    stats = {
        'total': 0,
        'correct': 0,
        'unknown_pred': 0,
        'per_class': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'errors': [],
    }

    for idx, sample in enumerate(tqdm(samples, desc='评测中')):
        query = sample['query']
        gt_response = sample['response']
        images = sample.get('images', [])

        gt_id, _ = extract_behavior_id(gt_response)
        if gt_id == 'unknown':
            continue

        try:
            pred_text = inference_vllm(args, query, images)
        except Exception as e:
            stats['errors'].append({'sample_idx': idx, 'error': str(e)})
            continue

        if args.debug and stats['total'] < 3:
            print(f'\n--- 样本 {stats["total"]+1} ---')
            print(f'GT ID: {gt_id}')
            print(f'图片: {images[0] if images else "N/A"}')
            print(f'模型输出: {pred_text[:500]}')

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

    if stats['errors']:
        print(f'\n前 5 个错误样本:')
        for i, err in enumerate(stats['errors'][:5]):
            print(f'  [{i+1}] GT: {err.get("gt_id", "?")} | Pred: {err.get("pred_id", "?")} | Image: {err.get("image", "N/A")}')

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
