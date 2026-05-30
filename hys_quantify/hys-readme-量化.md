## 环境准备

```bash
conda create -n llmpress python=3.10 -y
conda activate llmpress
pip install llmcompressor
pip install ms-swift
```

## 运行

```bash
CUDA_VISIBLE_DEVICES=0 python quantize_awq.py \
    --model /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b \
    --output /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-awq \
    --bits 4 --group_size 128 \
    --dataset /media/ddc/新加卷/hys/qmy/agent/data/sft_val.jsonl \
    --copy_config --official_model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B
```

## 量化参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--bits` | 4 | 权重量化位数（W4 = 4-bit） |
| `--group_size` | 128 | 每 128 个权重共享一个缩放因子 |
| `--dataset` | sft_val.jsonl | 校准数据集（用于观察激活分布） |

## 4-bit vs INT4

### 4-bit（泛指）

所有用 4 位存储的量化方式，包括：

| 类型 | 全称 | 特点 |
|------|------|------|
| INT4 | Integer 4-bit | 整数，均匀分布 |
| FP4 | Float 4-bit | 浮点数，有指数位 |
| NF4 | NormalFloat 4-bit | 基于正态分布的浮点数（bitsandbytes 用） |
| MXFP4 | Microscaling FP4 | 分块缩放的浮点数 |

### INT4（你的场景）

- AWQ 使用的是 INT4 整数量化
- 量化级别：-8, -7, ..., -1, 0, 1, ..., 7（共 16 个整数）
- 分布：均匀（每个级别间距相等）
- 对应 scheme：`W4A16`（Weight 4-bit INT，Activation 16-bit）

### 对比

| 特性 | INT4 | NF4 |
|------|------|-----|
| 分布 | 均匀 | 非均匀（匹配正态分布） |
| 适用场景 | 通用 | 权重近似正态分布时 |
| 精度 | 中 | 高 |
| 使用框架 | AWQ, GPTQ | bitsandbytes |