# 量化工具集

> 支持 BnB NF4、GPTQ、AWQ 三种量化方法，适配 Qwen3-VL 多模态模型

## 一句话总结

> **快速实验/QLoRA 用 BnB，Qwen3-VL 生产部署用 GPTQ（推荐），Qwen2-VL 用 AWQ。**

---

## ⚠️ Qwen3-VL 量化方案选择

| 方案 | 推荐度 | 说明 |
|------|--------|------|
| **GPTQ** | ⭐⭐⭐ **强烈推荐** | GPTQModel 原生支持 Qwen3-VL，同时量化 LLM + ViT |
| **BnB NF4** | ⭐⭐ 推荐 | 最简单，仅量化 LLM，ViT 保持 FP16 |
| **AWQ (llmcompressor)** | ⭐ 可用 | 忽略视觉编码器，仅量化 LLM |
| **AWQ (AutoAWQ)** | ⚠️ 实验性 | AutoAWQ 不支持 Qwen3-VL，已弃用 |

**关键原因**：
- Qwen3-VL 使用 `Qwen3VLForConditionalGeneration` 架构（`model_type: qwen3_vl`）
- Qwen3-VL 有 **DeepStack** 特性（融合 ViT 第 5、11、17 层特征），架构与 Qwen2-VL 不同
- AutoAWQ 官方不支持 Qwen3-VL，已被弃用
- GPTQModel 原生支持 Qwen3-VL 多模态架构

---

## 快速选择

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 快速实验 | **BnB** | 一行代码，分钟级完成 |
| QLoRA 微调 | **BnB** | 原生支持，显存节省 60%+ |
| Qwen3-VL 部署 | **GPTQ** | 原生支持，同时量化 LLM + ViT |
| Qwen2-VL 部署 | **AWQ** | AutoAWQ 原生支持 |
| vLLM 部署 | **GPTQ / AWQ** | 均支持 vLLM |

---

## 环境搭建

### BnB 环境（最简单）

```bash
conda create -n bnb python=3.10 -y
conda activate bnb
pip install bitsandbytes accelerate peft
```

### GPTQ 环境（推荐 Qwen3-VL）

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

### GPTQ 量化（推荐 Qwen3-VL）

```bash
conda activate swifthys
cd GPTQ
bash run_gptq.sh
```

### AWQ 量化（Qwen2-VL 推荐）

```bash
conda activate llmpress
cd AWQ
bash run.sh
```

> **注意**：Qwen3-VL 使用 AWQ 时，视觉编码器不会被量化（仅量化 LLM）。推荐使用 GPTQ。

---

## 模型加载方式

### Qwen3-VL 加载（必须使用专用类）

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# 正确方式
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/path/to/Qwen3-VL-4B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("/path/to/Qwen3-VL-4B-Instruct")

# 错误方式（会报错）
# model = AutoModelForCausalLM.from_pretrained(...)  # ❌
# model = Qwen3_5ForConditionalGeneration.from_pretrained(...)  # ❌
```

### Qwen2-VL 加载

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/path/to/Qwen2-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
```

---

## vLLM 部署

### GPTQ 模型（推荐 Qwen3-VL）

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

---

## 性能参考（Qwen3-VL-4B）

| 指标 | FP16 | GPTQ 4-bit | AWQ 4-bit | BnB NF4 |
|------|------|------------|-----------|---------|
| 模型大小 | ~4.4 GB | ~1.1 GB | ~1.1 GB | ~2.2 GB |
| 显存占用 | ~8 GB | ~1.6 GB | ~2 GB | ~4 GB |
| 推理速度 | 20 tok/s | 47 tok/s | 45 tok/s | 15 tok/s |
| 精度损失 | - | < 2% | < 1% | < 1% |
| 多模态支持 | ✅ | ✅ LLM+ViT | ⚠️ 仅 LLM | ⚠️ 仅 LLM |

---

## 常见问题

### Q: Qwen3-VL 应该用哪个量化方法？

**A: 推荐 GPTQ**。GPTQModel 原生支持 Qwen3-VL 多模态架构，同时量化语言模型和视觉编码器。

### Q: AutoAWQ 支持 Qwen3-VL 吗？

**A: 不支持**。AutoAWQ 官方仅支持 Qwen2-VL/Qwen2.5-VL，且已被弃用。如需使用 AWQ，请用 llmcompressor。

### Q: 量化后模型如何加载？

**A: 必须使用专用模型类**：
```python
# Qwen3-VL
from transformers import Qwen3VLForConditionalGeneration
model = Qwen3VLForConditionalGeneration.from_pretrained(...)

# Qwen2-VL
from transformers import Qwen2VLForConditionalGeneration
model = Qwen2VLForConditionalGeneration.from_pretrained(...)
```

### Q: transformers 版本要求？

**A:** Qwen3-VL 需要 `transformers >= 4.57.0`（从源码安装）：
```bash
pip install git+https://github.com/huggingface/transformers
```

---

## 相关链接

- [Qwen3-VL-4B-Instruct 模型](https://www.modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct)
- [AutoAWQ（已弃用）](https://github.com/casper-hansen/AutoAWQ)
- [GPTQModel](https://github.com/ModelCloud/GPTQModel)
- [llmcompressor](https://github.com/vllm-project/llm-compressor)
- [vLLM 文档](https://docs.vllm.ai/)
