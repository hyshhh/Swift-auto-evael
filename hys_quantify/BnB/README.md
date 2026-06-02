# BnB NF4 量化（支持 vLLM INT4 部署）

> bitsandbytes NormalFloat 4-bit，基于正态分布的 4-bit 量化

## 一句话总结

> **BnB 是最简单的量化方案，一行代码完成，适合快速实验、QLoRA 微调和 vLLM 部署。**

---

## 快速开始

```bash
# 1. 环境准备
conda create -n bnb python=3.10 -y
conda activate bnb
pip install bitsandbytes accelerate peft

# 2. 量化
bash run.sh

# 3. 验证
python verify_bnb.py --model /path/to/output --bits 4

# 4. vLLM 部署
vllm serve /path/to/output \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port 7890
```

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `quantize_bnb.py` | BnB NF4/INT8 量化脚本（自动修补 vLLM 兼容配置） |
| `verify_bnb.py` | 量化模型验证（支持多模态图像/视频） |
| `config_bnb_qlora_sft.json` | QLoRA 微调配置 |
| `run.sh` | 一键启动脚本（Qwen3-VL-4B） |

---

## 量化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | （必填） | 原始模型路径 |
| `--output` | （必填） | 输出模型路径 |
| `--bits` | 4 | 量化位数 (4=NF4, 8=INT8) |
| `--double_quant` | False | 双重量化（额外节省 ~5%） |
| `--compute_dtype` | bfloat16 | 计算精度 |
| `--copy_config` | False | 从官方模型复制配置文件 |
| `--official_model` | None | 官方模型路径 |

---

## vLLM INT4 部署（重点）

参考 [vLLM INT4 量化文档](https://docs.vllm.com.cn/en/latest/features/quantization/int4/)，BnB NF4 模型可通过 vLLM 部署。

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--quantization` | `bitsandbytes` | 告诉 vLLM 使用 BnB 量化 |
| `--load-format` | `bitsandbytes` | 指定权重加载格式 |
| `--dtype` | `bfloat16` | 计算精度 |

### 部署命令

```bash
# 基础部署
vllm serve /path/to/Qwen3-VL-4B-BnB-NF4 \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port 7890 \
    --api-key abc123 \
    --served-model-name Qwen/Qwen3-VL-4B-BnB

# 完整部署（带工具调用支持）
vllm serve /path/to/Qwen3-VL-4B-BnB-NF4 \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --dtype bfloat16 \
    --max-model-len 10240 \
    --port 7890 \
    --api-key abc123 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 10 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
```

### 测试推理

```bash
curl http://localhost:7890/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer abc123" \
  -d '{
    "model": "Qwen/Qwen3-VL-4B-BnB",
    "messages": [{"role": "user", "content": "你好，请介绍一下你自己"}],
    "max_tokens": 512
  }'
```

---

## 性能指标

| 指标 | FP16 | BnB NF4 | BnB INT8 |
|------|------|---------|----------|
| 模型大小 | ~8 GB | ~2.5 GB | ~4.5 GB |
| 显存占用 | ~16 GB | ~4 GB | ~6 GB |
| 推理速度 | 快 | 中等 | 较快 |
| 精度损失 | - | < 1% | < 0.5% |

---

## NF4 vs INT8

| 特性 | NF4 | INT8 |
|------|-----|------|
| 压缩率 | 75% | 50% |
| 显存节省 | ~60% | ~40% |
| 精度 | 高 | 极高 |
| 推荐场景 | 显存受限 / vLLM 部署 | 追求精度 |

---

## 多模态支持

BnB 对 Qwen3-VL 多模态模型支持非常好：

```
量化策略：
├── 语言模型 (LLM)  → NF4 量化（压缩 75%）
└── 视觉编码器 (VE) → FP16 保持（精度不变）

结果：文本+图像+视频推理均正常 ✓
```

---

## QLoRA 微调

BnB 是 QLoRA 微调的最佳搭配：

```bash
# 使用 ms-swift 微调
swift sft \
    --model /path/to/Qwen3-VL-4B-BnB-NF4 \
    --dataset /path/to/dataset.jsonl \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --num_train_epochs 3 \
    --learning_rate 2e-4 \
    --bf16 true
```

---

## 常见问题

**Q: vLLM 加载报错 `ValueError: No supported config format found`？**
A: 确保同时指定 `--quantization bitsandbytes` 和 `--load-format bitsandbytes`。

**Q: 为什么比 GPTQ 慢？**
A: BnB 需要实时反量化，GPTQ 预计算优化。如果追求推理速度，推荐 GPTQ/AWQ。

**Q: 双重量化值得开吗？**
A: 值得！额外节省 ~5% 显存，精度几乎无损。

**Q: 量化后 config.json 里没有 quantization_config？**
A: 本脚本会自动修补 config.json，添加 vLLM 所需的 `quantization_config` 字段。如果仍有问题，使用 `--copy_config --official_model` 参数。

---

## 相关链接

- [vLLM INT4 量化文档](https://docs.vllm.com.cn/en/latest/features/quantization/int4/)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [HuggingFace 量化](https://huggingface.co/docs/transformers/quantization/overview)
