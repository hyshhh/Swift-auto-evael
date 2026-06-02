# 量化工具集

> 支持 BnB NF4、GPTQ、AWQ 三种量化方法，适配 Qwen3-VL 多模态模型

## 一句话总结

> **快速实验/QLoRA 用 BnB，生产部署用 GPTQ/AWQ。**

---

## 快速选择

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 快速实验 | **BnB** | 一行代码，分钟级完成 |
| QLoRA 微调 | **BnB** | 原生支持，显存节省 60%+ |
| vLLM 部署 | **BnB / GPTQ** | 均支持 vLLM INT4 推理 |
| 生产部署 | **GPTQ / AWQ** | 推理速度快，显存效率高 |

---

## 环境搭建

### BnB 环境（最简单）

```bash
conda create -n bnb python=3.10 -y
conda activate bnb
pip install bitsandbytes accelerate peft
```

### GPTQ 环境（推荐）

```bash
conda create -n swifthys python=3.10 -y
conda activate swifthys
pip install ms-swift gptqmodel optimum accelerate
```

### AWQ 环境

```bash
conda create -n llmpress python=3.10 -y
conda activate llmpress
pip install llmcompressor ms-swift

# 如需使用 AutoAWQ（仅支持 Qwen2-VL/Qwen2.5-VL，不支持 Qwen3-VL）
pip install autoawq
```

---

## 使用方法

### BnB NF4 量化（最简单，支持 vLLM）

```bash
conda activate bnb
cd BnB
bash run.sh
```

### GPTQ 量化（推荐生产）

```bash
conda activate swifthys
cd GPTQ
bash run_gptq.sh
```

### AWQ 量化（Qwen3-VL 推荐）

```bash
conda activate llmpress
cd AWQ
bash run.sh
```

> **注意**：Qwen3-VL 请使用 llmcompressor（默认选项），AutoAWQ 不支持 Qwen3-VL。

---

## vLLM 部署

### BnB NF4 模型

```bash
vllm serve /path/to/model-bnb-nf4 \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port 7890 \
    --api-key abc123
```

### GPTQ 模型

```bash
vllm serve /path/to/model-gptq \
    --quantization gptq \
    --dtype float16 \
    --max-model-len 8192 \
    --port 7890 \
    --api-key abc123
```

### AWQ 模型

```bash
vllm serve /path/to/model-awq \
    --quantization awq \
    --max-model-len 8192 \
    --port 7890 \
    --api-key abc123
```

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
| **vLLM 部署** | ✅ bitsandbytes | ✅ gptq | ✅ awq |

---

## 目录结构

```
hys_quantify/
├── README.md                # 本文档
├── README-量化方法.md        # 详细量化方法说明
├── BnB/                     # BnB NF4 量化
│   ├── quantize_bnb.py      # 量化脚本（自动修补 vLLM 兼容配置）
│   ├── verify_bnb.py        # 验证脚本（支持多模态）
│   ├── run.sh               # 启动脚本
│   ├── config_bnb_qlora_sft.json  # QLoRA 配置
│   └── README.md
├── GPTQ/                    # GPTQ 量化
│   ├── quantize_gptq.py     # 量化脚本
│   ├── run_gptq_swift.py    # ms-swift 量化
│   ├── run_gptq.sh          # 启动脚本
│   └── README.md
└── AWQ/                     # AWQ 量化
    ├── quantize_awq.py      # llmcompressor 量化
    ├── quantize_autoawq.py  # AutoAWQ 量化
    ├── run.sh               # 启动脚本
    └── README.md
```

---

## 相关资源

- [vLLM INT4 量化文档](https://docs.vllm.com.cn/en/latest/features/quantization/int4/)
- [BnB (bitsandbytes)](https://github.com/TimDettmers/bitsandbytes)
- [GPTQModel](https://github.com/ModelCloud/GPTQModel)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
