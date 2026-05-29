# MS-Swift 训练与推理笔记

## 1. 基础推理（LoRA 动态叠加）

使用 Transformers 引擎，LoRA 权重动态叠加到基座模型，适合快速测试。

```bash
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
    --adapters /media/ddc/新加卷/hys/qmy/ms-swift/output/v2-20260528-211924/checkpoint-31 \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
```

## 2. 合并 LoRA 权重

将 LoRA 权重合并到基座模型，生成完整模型。

```bash
CUDA_VISIBLE_DEVICES=2 \
swift export \
    --model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
    --adapters /media/ddc/新加卷/hys/qmy/ms-swift/output/v2-20260528-211924/checkpoint-31 \
    --merge_lora true \
    --output_dir /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b
```

## 3. 修复合并后的模型配置

合并后需要复制缺失的配置文件，并删除多余文件。

### 3.1 复制缺失的文件

```bash
# 复制官方模型的配置文件
cp /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B/config.json /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/
cp /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B/video_preprocessor_config.json /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/

# 复制 tokenizer 相关文件（可选，但推荐）
cp /media/ddc/新加卷/hys/hysnew/Qwen3.5-2B-AWQ/tokenizer_config.json /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/
cp /media/ddc/新加卷/hys/hysnew/Qwen3.5-2B-AWQ/vocab.json /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/
cp /media/ddc/新加卷/hys/hysnew/Qwen3.5-2B-AWQ/merges.txt /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/
```

### 3.2 删除多余的文件

```bash
rm /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/processor_config.json
rm /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/args.json
```

## 4. 使用 VLLM 推理合并后的模型

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b \
    --api-key abc123 \
    --served-model-name Qwen/Qwen3-VL-4B-AWQ \
    --max-model-len 10240 \
    --port 7890 \
    --gpu-memory-utilization 0.15 \
    --max-num-seqs 10 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
```

## 5. SFT 训练命令

### 5.1 环境准备

```bash
conda activate swift
```

### 5.2 训练命令

```bash
CUDA_VISIBLE_DEVICES=1 \
swift sft \
    --model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
    --tuner_type lora \
    --dataset /media/ddc/新加卷/hys/qmy/agent/data/sft_train.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 2 \
    --lora_alpha 8 \
    --target_modules all-linear \
    --gradient_accumulation_steps 32 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 6 \
    --model_author swift \
    --model_name swift-robot
```

## 6. 关键参数说明

### 6.1 数据加载参数

| 参数 | 说明 |
|------|------|
| `--dataloader_num_workers 4` | 使用 4 个子进程并行加载数据 |
| `--num_train_epochs 1` | 整个数据集遍历训练 1 轮 |
| `--per_device_train_batch_size 1` | 每张 GPU 一次处理 1 条数据 |

### 6.2 LoRA 参数

| 参数 | 说明 |
|------|------|
| `--lora_rank 8` | LoRA 的秩（矩阵维度），越大拟合能力越强，但显存占用越多 |
| `--lora_alpha 32` | 缩放系数，控制 LoRA 权重对原始模型的影响程度 |
| `--target_modules all-linear` | 对所有线性层应用 LoRA，可减少到 `q_proj,v_proj` 或 `q_proj,k_proj,v_proj,o_proj` |

### 6.3 多模态模型冻结参数

```bash
--freeze_llm true           # 冻结 LLM 部分
--freeze_vit true           # 冻结 ViT 部分
--freeze_aligner true       # 冻结对齐器
--modules_to_save embed_tokens lm_head  # 即使使用 LoRA，也对这两个层进行全参数训练并保存权重
```

### 6.4 训练策略参数

| 参数 | 说明 |
|------|------|
| `--gradient_accumulation_steps 16` | 每 16 步才更新一次模型权重 |
| `--torch_dtype bfloat16` | 加载模型时，把权重从 float32 转换为 bfloat16 |

### 6.5 总 Step 数计算公式

```
总 step 数 = 数据量 ÷ (per_device_train_batch_size × gradient_accumulation_steps × GPU数) × epochs
```

## 7. 训练指标说明

| 指标 | 值 | 说明 |
|------|-----|------|
| loss | 0.3411 | 损失值，越低越好，0.34 说明模型在快速学习 |
| grad_norm | 7.841 | 梯度范数，用于监控梯度爆炸/消失，正常范围 |
| learning_rate | 5e-05 | 当前学习率，因为 warmup 还没结束，从 0 逐渐增加到 1e-4 |
| token_acc | 0.915 | token 准确率 91.5%，模型预测正确的 token 比例 |
| epoch | 0.03252 | 当前进度 3.25%，数据集还没遍历完一遍 |
| global_step/max_steps | 1/31 | 第 1 步 / 共 31 步 |
| elapsed_time | 41s | 已训练 41 秒 |
| remaining_time | 20m 35s | 预计剩余 20 分 35 秒 |
| memory(GiB) | 10.7 | 显存占用 10.7 GiB |
| train_speed(s/it) | 41.16 | 每步耗时 41.16 秒 |

## 8. 行为识别评测脚本（eval_behavior.py）

使用 vLLM API 模式进行行为识别评测。

### 8.1 脚本功能

- 通过 vLLM API 接口调用模型进行推理
- 支持多模态输入（图片 + 文本）
- 自动提取模型输出的 behavior_id 并与 ground truth 对比
- 输出评测结果和错误样本分析

### 8.2 使用方法

```bash
python eval_behavior.py \
    --vllm_url http://localhost:7890 \
    --model_name Qwen/Qwen3-VL-4B-AWQ \
    --dataset /media/ddc/新加卷/hys/qmy/agent/data/sft_val.jsonl \
    --debug
```

### 8.3 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vllm_url` | http://localhost:7890 | vLLM 服务地址 |
| `--model_name` | Qwen/Qwen3-VL-4B-AWQ | vLLM 模型名称 |
| `--api_key` | abc123 | API 密钥 |
| `--dataset` | （必填） | 评测数据集 jsonl 路径 |
| `--max_new_tokens` | 256 | 最大生成 token 数 |
| `--debug` | false | 打印前几条的原始输出 |
| `--output` | 自动命名 | 预测结果保存路径 |

### 8.4 数据集格式

数据集为 jsonl 格式，每行一个 JSON 对象：

```json
{
    "query": "请识别视频中的行为",
    "response": "{\"behavior_id\": \"1\", \"description\": \"...\"}",
    "images": ["/path/to/image1.jpg"]
}
```

### 8.5 输出结果

评测结果保存为 JSON 文件，包含：

```json
{
    "accuracy": 0.85,
    "total": 100,
    "correct": 85,
    "unknown_pred": 5,
    "per_class": {
        "1": {"total": 30, "correct": 28},
        "2": {"total": 25, "correct": 20}
    },
    "error_samples": [...]
}
```

### 8.6 脚本代码

```python
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
```