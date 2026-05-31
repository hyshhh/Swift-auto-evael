## 环境准备

```bash
conda create -n llmpress python=3.10 -y
conda activate llmpress
pip install llmcompressor
pip install ms-swift

# GPTQ 量化依赖（推荐，原生支持 Qwen3.5）
pip install gptqmodel optimum accelerate
```

## 运行

```bash
CUDA_VISIBLE_DEVICES=0 python quantize_awq.py \
    --model /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b \
    --output /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-awq \
    --bits 4 --group_size 128 \
    --dataset /media/ddc/新加卷/hys/qmy/agent/data/sft_val.jsonl \
    --copy_config --official_model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B

python quantize_autoawq.py \
    --model /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b \
    --output /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-awq \
    --dataset /media/ddc/新加卷/hys/qmy/agent/data/sft_val.jsonl \
    --bits 4 --group_size 128 \
    --copy_config --official_model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
    --gpu 1
```

### GPTQ 量化（推荐，原生支持 Qwen3.5）

```bash
python quantize_gptq.py \
    --model /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b \
    --output /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-gptq \
    --dataset /media/ddc/新加卷/hys/qmy/agent/data/sft_val.jsonl \
    --bits 4 --group_size 128 \
    --copy_config --official_model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
    --gpu 0
```

或使用运行脚本：

```bash
bash run_quantize_gptq.sh
```

## vLLM 推理服务

### AWQ 模型

```bash
CUDA_VISIBLE_DEVICES=2 vllm serve /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-awq \
    --api-key abc123 \
    --served-model-name Qwen/Qwen3-VL-4B-AWQ \
    --max-model-len 10240 \
    --port 7890 \
    --gpu-memory-utilization 0.15 \
    --max-num-seqs 10 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
```

### GPTQ 模型

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-gptq \
    --api-key abc123 \
    --served-model-name Qwen/Qwen3.5-2B-GPTQ \
    --max-model-len 10240 \
    --port 7890 \
    --gpu-memory-utilization 0.15 \
    --max-num-seqs 10 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
```
## 量化参数说明

### AWQ 量化参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--bits` | 4 | 权重量化位数（W4 = 4-bit） |
| `--group_size` | 128 | 每 128 个权重共享一个缩放因子 |
| `--dataset` | sft_val.jsonl | 校准数据集（用于观察激活分布） |

### GPTQ 量化参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--bits` | 4 | 权重量化位数（2/3/4/5/6/8） |
| `--group_size` | 128 | 每 128 个权重共享一个缩放因子 |
| `--sym` | True | 对称量化 |
| `--desc_act` | True | 激活排序（提升精度） |
| `--true_sequential` | True | 真正顺序量化 |
| `--damp_percent` | 0.01 | Hessian 阻尼系数 |
| `--batch_size` | 1 | 量化批次大小 |
| `--lm_head` | False | 是否量化 lm_head |

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

## AWQ vs GPTQ 对比

| 特性 | AWQ (AutoAWQ) | GPTQ (GPTQModel) |
|------|---------------|------------------|
| **Qwen3.5 支持** | ❌ 需要 monkey-patch | ✅ 原生支持（v5.8.0+） |
| **量化原理** | 激活感知权重量化 | 基于 OBS 的逐层量化 |
| **校准数据** | 文本列表 | 文本列表（自动 tokenize） |
| **vLLM 兼容** | ✅ | ✅ |
| **维护状态** | 更新较慢 | 积极维护（v7.0.0） |
| **推荐场景** | 传统模型 | 新架构（Qwen3.5 等） |