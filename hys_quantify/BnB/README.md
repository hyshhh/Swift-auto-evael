# BnB NF4 量化

> bitsandbytes NormalFloat 4-bit，基于正态分布的 4-bit 量化

## 一句话总结

> **BnB 是最简单的量化方案，一行代码完成，适合快速实验和 QLoRA 微调。**

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
python verify.py --model /path/to/output
```

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `quantize_bnb.py` | BnB NF4/INT8 量化脚本 |
| `verify_bnb.py` | 量化模型验证（支持多模态） |
| `config_bnb_qlora_sft.json` | QLoRA 微调配置 |
| `run.sh` | 一键启动脚本 |

---

## 量化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | （必填） | 原始模型路径 |
| `--output` | （必填） | 输出模型路径 |
| `--bits` | 4 | 量化位数 (4=NF4, 8=INT8) |
| `--double_quant` | False | 双重量化（额外节省 ~5%） |
| `--compute_dtype` | bfloat16 | 计算精度 |

---

## 性能指标

| 指标 | FP16 | BnB NF4 | BnB INT8 |
|------|------|---------|----------|
| 模型大小 | 4 GB | 2.1 GB | 2.8 GB |
| 显存占用 | 8 GB | 2.1 GB | 2.8 GB |
| 推理速度 | 20 tok/s | 15 tok/s | 20 tok/s |
| 精度损失 | - | < 1% | < 0.5% |

---

## NF4 vs INT8

| 特性 | NF4 | INT8 |
|------|-----|------|
| 压缩率 | 75% | 50% |
| 显存节省 | ~60% | ~40% |
| 精度 | 高 | 极高 |
| 推荐场景 | 显存受限 | 追求精度 |

---

## 多模态支持

BnB 对多模态模型支持非常好：

```
量化策略：
├── 语言模型 (LLM)  → NF4 量化（压缩 75%）
└── 视觉编码器 (VE) → FP16 保持（精度不变）

结果：文本+图像推理均正常 ✓
```

---

## QLoRA 微调

BnB 是 QLoRA 微调的最佳搭配：

```bash
# 使用 ms-swift 微调
swift sft \
    --model /path/to/model-bnb \
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

**Q: 为什么比 GPTQ 慢？**
A: BnB 需要实时反量化，GPTQ 预计算优化。

**Q: 能部署吗？**
A: 可以，但推荐 GPTQ/AWQ 用于生产。

**Q: 双重量化值得开吗？**
A: 值得！额外节省 ~5% 显存，精度几乎无损。

---

## 相关链接

- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [HuggingFace 量化](https://huggingface.co/docs/transformers/quantization/overview)
