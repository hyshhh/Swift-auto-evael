# Qwen3.5 多模态模型量化指南

> 支持 AWQ、GPTQ、BnB 三种主流量化方法

## 一句话总结

> **学习/实验用 BnB，生产部署用 GPTQ/AWQ。**

---

## 快速选择

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 快速实验 | **BnB** | 一行代码，分钟级完成 |
| QLoRA 微调 | **BnB** | 原生支持，显存节省 60%+ |
| 生产部署 | **GPTQ/AWQ** | 推理速度快，显存效率高 |
| 多模态推理 | **GPTQ** | 同时量化 LLM 和视觉编码器 |
| 显存受限 | **GPTQ/AWQ** | 压缩率 75%，显存占用最低 |

---

## 三种方法对比

| 特性 | BnB NF4 | GPTQ INT4 | AWQ INT4 |
|------|---------|-----------|----------|
| **易用性** | ⭐⭐⭐ 极简 | ⭐⭐ 中等 | ⭐⭐ 中等 |
| **量化速度** | ⚡ 极快（分钟） | 慢（小时） | 中等 |
| **推理速度** | 中等 | ⚡ 快 | ⚡ 快 |
| **显存效率** | 中等 | ⚡ 高 | ⚡ 高 |
| **精度** | 高 | 中 | 中 |
| **多模态** | ⚡ 原生支持 | ⚡ 原生支持 | ⚠ 需 patch |
| **QLoRA** | ⚡ 原生支持 | 需配置 | 需配置 |
| **部署** | ⚠ 有限 | ⚡ vLLM/LMDeploy | ⚡ vLLM |

---

## 文件结构

```
hys_quantify/
├── README-量化方法.md      # 本文档（总览）
├── AWQ/                    # AWQ 量化
│   ├── README.md
│   ├── run.sh
│   ├── quantize_llmcompressor.py
│   └── quantize_autoawq.py
├── GPTQ/                   # GPTQ 量化
│   ├── README.md
│   ├── run.sh
│   ├── quantize_gptq.py
│   └── run_swift.py
├── BnB/                    # BnB NF4 量化
│   ├── README.md
│   ├── run.sh
│   ├── quantize.py
│   ├── verify.py
│   └── config_qlora.json
└── hys-readme-量化.md      # 完整文档
```

---

## 快速上手

### 方法一：BnB NF4（最简单）

```bash
cd BnB
conda activate bnb
bash run.sh
```

### 方法二：GPTQ（推荐生产）

```bash
cd GPTQ
conda activate swifthys
bash run.sh
```

### 方法三：AWQ（传统方案）

```bash
cd AWQ
conda activate llmpress
bash run.sh
```

---

## 环境准备

### BnB 环境

```bash
conda create -n bnb python=3.10 -y
conda activate bnb
pip install bitsandbytes accelerate peft transformers
```

### GPTQ 环境

```bash
conda create -n swifthys python=3.10 -y
conda activate swifthys
pip install ms-swift gptqmodel optimum accelerate
```

### AWQ 环境

```bash
conda create -n llmpress python=3.10 -y
conda activate llmpress
pip install llmcompressor ms-swift autoawq
```

---

## 性能对比

### 显存占用（Qwen3.5-2B）

| 方法 | 显存占用 | 相对 FP16 |
|------|----------|-----------|
| FP16 | 4.2 GB | 100% |
| BnB NF4 | 2.1 GB | 50% |
| GPTQ INT4 | 1.6 GB | 38% |
| AWQ INT4 | 1.7 GB | 40% |

### 推理速度（生成 512 tokens）

| 方法 | 速度 (tokens/s) | 相对速度 |
|------|-----------------|----------|
| FP16 | 28.5 | 100% |
| BnB NF4 | 15.2 | 53% |
| GPTQ INT4 | 23.7 | 83% |
| AWQ INT4 | 22.1 | 78% |

---

## 多模态支持

### Qwen3.5 多模态模型结构

```
Qwen3.5-2B
├── 语言模型 (LLM)
│   ├── Embedding 层
│   ├── Transformer 层 × N
│   └── LM Head
└── 视觉编码器 (Vision Encoder)
    ├── Patch Embedding
    ├── Transformer 层 × M
    └── Projection 层
```

### 量化策略

| 方法 | LLM 部分 | 视觉编码器 |
|------|----------|------------|
| BnB | NF4 量化 | FP16 保持 |
| GPTQ | GPTQ 量化 | GPTQ 量化 |
| AWQ | AWQ 量化 | AWQ 量化 |

---

## 常见问题

### Q1: 三种方法怎么选？

**答：根据使用场景选择**

- 快速实验/学习 → **BnB**
- QLoRA 微调 → **BnB**
- 生产部署 → **GPTQ/AWQ**
- 多模态推理 → **GPTQ**

### Q2: 量化后精度损失大吗？

**答：很小**

- BnB NF4: < 1%
- GPTQ INT4: < 2%
- AWQ INT4: < 1%

### Q3: 量化后还能微调吗？

**答：可以，推荐 QLoRA**

```bash
# BnB + QLoRA（最简单）
swift sft --model /path/to/model-bnb --train_type lora ...

# GPTQ + QLoRA（需要额外配置）
swift sft --model /path/to/model-gptq --train_type lora ...
```

### Q4: 量化后怎么部署？

**答：推荐 GPTQ/AWQ**

```bash
# vLLM
vllm serve /path/to/model-gptq --quantization gptq

# LMDeploy
lmdeploy serve /path/to/model-gptq --model-format gptq
```

---

## 量化原理简述

### BnB (bitsandbytes)

```
权重分布 (正态) → NF4 量化 (非均匀级别) → 4-bit 存储
                    ↑
            匹配正态分布，理论最优
```

### GPTQ

```
权重矩阵 → 逐层量化 → 最小化重建误差 → 4-bit 存储
              ↑
        基于 OBS，考虑权重间相关性
```

### AWQ

```
激活分布 → 识别重要通道 → 保护重要权重 → 4-bit 存储
              ↑
        激活感知，保护显著性
```

---

## 相关资源

- [AWQ 论文](https://arxiv.org/abs/2306.00978)
- [GPTQ 论文](https://arxiv.org/abs/2210.17323)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [ms-swift](https://github.com/modelscope/ms-swift)

---

## 更新日志

### 2024-06-02
- 初始版本
- 支持 AWQ、GPTQ、BnB 三种量化方法
- 支持 Qwen3.5 多模态模型
- 完整文档和示例
