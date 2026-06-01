# MS-Swift 训练与推理笔记

## 1. 基础推理（LoRA 动态叠加）

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

```bash
CUDA_VISIBLE_DEVICES=2 \
swift export \
    --model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
    --adapters /media/ddc/新加卷/hys/qmy/ms-swift/output/v2-20260528-211924/checkpoint-31 \
    --merge_lora true \
    --output_dir /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b
```

## 3. 修复合并后的模型配置

```bash
# 复制官方模型的配置文件
cp /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B/config.json /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/
cp /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B/video_preprocessor_config.json /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/

# 复制 tokenizer 相关文件（可选，但推荐）
cp /media/ddc/新加卷/hys/hysnew/Qwen3.5-2B-AWQ/tokenizer_config.json /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/
cp /media/ddc/新加卷/hys/hysnew/Qwen3.5-2B-AWQ/vocab.json /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/
cp /media/ddc/新加卷/hys/hysnew/Qwen3.5-2B-AWQ/merges.txt /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/

# 删除多余的文件
rm /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/processor_config.json
rm /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/args.json
```

## 4. 使用 VLLM 推理

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

## 5. 评测

```bash
python eval_behavior.py \
    --vllm_url http://localhost:7890 \
    --model_name Qwen/Qwen3-VL-4B-AWQ \
    --dataset /media/ddc/新加卷/hys/qmy/agent/data/sft_val.jsonl \
    --debug
```

## 6. SFT 训练

```bash
conda activate swift

CUDA_VISIBLE_DEVICES=1 \
swift sft \
    --model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
    --quant_bits 4 \
    --tuner_type lora \
    --dataset /media/ddc/新加卷/hys/qmy/agent/data/sft_train.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 4 \
    --freeze_llm true \
    --freeze_vit true \
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

## 7. AWQ 量化

```bash
# 激活环境
conda activate llmpress

# 安装依赖
pip install llmcompressor
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 运行量化
cd /media/ddc/新加卷/hys/hysnew3/Swift-auto-evael/hys_quantify

CUDA_VISIBLE_DEVICES=0 python quantize_awq.py \
    --model /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b \
    --output /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-awq \
    --bits 4 \
    --group_size 128 \
    --dataset alpaca \
    --copy_config \
    --official_model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B
```

## 8. GPTQ 量化

```bash
# 激活环境
conda activate swifthys

# 安装依赖（如未安装）
pip install gptqmodel optimum accelerate

# 运行量化（使用 gptqmodel 直接量化，支持 Qwen3.5 多模态）
cd /media/ddc/新加卷/hys/hysnew3/Swift-auto-evael/hys_quantify

python run_gptq_swift.py
```

> **说明**：`run_gptq_swift.py` 会同时量化语言模型和视觉编码器，并自动复制所有配置文件。

### GPTQ 模型推理

```bash
# 激活环境
conda activate vllm

CUDA_VISIBLE_DEVICES=2 vllm serve /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-gptq \
    --api-key abc123 \
    --served-model-name Qwen/Qwen3.5-2B-GPTQ \
    --max-model-len 10240 \
    --port 7890 \
    --quantization gptq \
    --dtype float16 \
    --gpu-memory-utilization 0.15
```

---

## 参数说明

### 数据加载参数

| 参数 | 说明 |
|------|------|
| `--dataloader_num_workers 4` | 使用 4 个子进程并行加载数据 |
| `--num_train_epochs 1` | 整个数据集遍历训练 1 轮 |
| `--per_device_train_batch_size 1` | 每张 GPU 一次处理 1 条数据 |

### LoRA 参数

| 参数 | 说明 |
|------|------|
| `--lora_rank 8` | LoRA 的秩（矩阵维度），越大拟合能力越强，但显存占用越多 |
| `--lora_alpha 32` | 缩放系数，控制 LoRA 权重对原始模型的影响程度 |
| `--target_modules all-linear` | 对所有线性层应用 LoRA |

### 多模态模型冻结参数

```bash
--freeze_llm true           # 冻结 LLM 部分
--freeze_vit true           # 冻结 ViT 部分
--freeze_aligner true       # 冻结对齐器
--modules_to_save embed_tokens lm_head  # 即使使用 LoRA，也对这两个层进行全参数训练并保存权重
```

### 训练策略参数

| 参数 | 说明 |
|------|------|
| `--gradient_accumulation_steps 16` | 每 16 步才更新一次模型权重 |
| `--torch_dtype bfloat16` | 加载模型时，把权重从 float32 转换为 bfloat16 |

### 总 Step 数计算公式

```
总 step 数 = 数据量 ÷ (per_device_train_batch_size × gradient_accumulation_steps × GPU数) × epochs
```

---

## 训练指标说明

| 指标 | 值 | 说明 |
|------|-----|------|
| loss | 0.3411 | 损失值，越低越好 |
| grad_norm | 7.841 | 梯度范数，用于监控梯度爆炸/消失 |
| learning_rate | 5e-05 | 当前学习率 |
| token_acc | 0.915 | token 准确率 91.5% |
| epoch | 0.03252 | 当前进度 3.25% |
| global_step/max_steps | 1/31 | 第 1 步 / 共 31 步 |
| elapsed_time | 41s | 已训练 41 秒 |
| remaining_time | 20m 35s | 预计剩余 20 分 35 秒 |
| memory(GiB) | 10.7 | 显存占用 10.7 GiB |
| train_speed(s/it) | 41.16 | 每步耗时 41.16 秒 |

---

## 评测脚本说明

### 脚本功能

- 通过 vLLM API 接口调用模型进行推理
- 支持多模态输入（图片 + 文本）
- 自动提取模型输出的 behavior_id 并与 ground truth 对比
- 输出评测结果和错误样本分析

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vllm_url` | http://localhost:7890 | vLLM 服务地址 |
| `--model_name` | Qwen/Qwen3-VL-4B-AWQ | vLLM 模型名称 |
| `--api_key` | abc123 | API 密钥 |
| `--dataset` | （必填） | 评测数据集 jsonl 路径 |
| `--max_new_tokens` | 256 | 最大生成 token 数 |
| `--debug` | false | 打印前几条的原始输出 |
| `--output` | 自动命名 | 预测结果保存路径 |

### 数据集格式

```json
{
    "query": "请识别视频中的行为",
    "response": "{\"behavior_id\": \"1\", \"description\": \"...\"}",
    "images": ["/path/to/image1.jpg"]
}
```

### 输出结果

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
