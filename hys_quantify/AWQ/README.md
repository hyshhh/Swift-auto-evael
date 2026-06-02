# AWQ 量化

> Activation-aware Weight Quantization，激活感知权重量化

## 一句话总结

> **AWQ 通过保护重要权重通道，实现高精度 4-bit 量化。推荐用于 Qwen2-VL/Qwen2.5-VL，Qwen3-VL 推荐使用 GPTQ。**

---

## ⚠️ 重要说明

### Qwen3-VL 量化方案选择

| 模型 | 推荐方案 | 原因 |
|------|----------|------|
| **Qwen3-VL** | **GPTQ** | GPTQModel 原生支持，同时量化 LLM + ViT |
| **Qwen3-VL** | BnB NF4 | 最简单，仅量化 LLM |
| **Qwen3-VL** | AWQ (llmcompressor) | 可用，但仅量化 LLM |
| **Qwen2-VL** | AWQ (AutoAWQ) | AutoAWQ 原生支持 |
| **Qwen2.5-VL** | AWQ (AutoAWQ) | AutoAWQ 原生支持 |

### AutoAWQ 已被官方弃用

- **AutoAWQ 已停止维护**，最后测试配置：Torch 2.6.0 + Transformers 4.51.3
- **推荐替代方案**：[llmcompressor](https://github.com/vllm-project/llm-compressor) 或 [GPTQModel](https://github.com/ModelCloud/GPTQModel)
- **Qwen3-VL 量化**：请使用 `quantize_gptq.py`（GPTQModel）

### AutoAWQ 官方支持的多模态模型

| 模型 | 状态 |
|------|------|
| Qwen2-VL | ✅ 官方支持 |
| Qwen2.5-VL | ✅ 官方支持 |
| Qwen2.5-Omni | ✅ 官方支持 |
| LLaVA / LLaVA-Next | ✅ 官方支持 |
| Phi3-V | ✅ 官方支持 |
| **Qwen3-VL** | ❌ **不支持**（架构不同，有 DeepStack 特性） |

---

## 快速开始

```bash
# 1. 环境准备
conda create -n awq python=3.10 -y
conda activate awq
pip install llmcompressor ms-swift

# 2. 量化（使用 llmcompressor）
bash run.sh

# 3. 验证
python verify.py --model /path/to/output
```

---

## 文件说明

| 文件 | 用途 | 推荐度 |
|------|------|--------|
| `quantize_awq.py` | llmcompressor 量化（支持多模态） | ⭐⭐⭐ 推荐 |
| `quantize_autoawq.py` | AutoAWQ 量化（仅支持 Qwen2-VL/Qwen2.5-VL） | ⚠️ 实验性 |
| `run.sh` | 一键启动脚本（默认使用 llmcompressor） | - |
| `run_llmcompressor.sh` | llmcompressor 直接启动脚本 | - |

---

## 量化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | （必填） | 原始模型路径 |
| `--output` | （必填） | 输出模型路径 |
| `--bits` | 4 | 量化位数（3/4） |
| `--group_size` | 128 | 分组大小 |
| `--dataset` | None | 校准数据集路径 |

---

## Qwen3-VL 量化（使用 llmcompressor）

```bash
CUDA_VISIBLE_DEVICES=0 python quantize_awq.py \
    --model /path/to/Qwen3-VL-4B-Instruct \
    --output /path/to/Qwen3-VL-4B-AWQ \
    --bits 4 \
    --group_size 128 \
    --dataset /path/to/calibration.jsonl \
    --copy_config \
    --official_model /path/to/Qwen3-VL-4B-Instruct
```

### 量化策略

```
Qwen3-VL 结构（使用 llmcompressor）：
├── 语言模型 (LLM)      → AWQ 4-bit 量化
├── 视觉编码器 (ViT)    → FP16 保持（忽略）
└── Projection 层       → AWQ 4-bit 量化

注意：AWQ 不量化视觉编码器，如需同时量化 LLM + ViT，请使用 GPTQ
```

---

## Qwen2-VL 量化（使用 AutoAWQ）

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'Qwen/Qwen2-VL-7B-Instruct'
quant_path = 'Qwen2-VL-7B-Instruct-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

---

## 性能指标

| 指标 | FP16 | AWQ 4-bit |
|------|------|-----------|
| 模型大小 | 4 GB | 1 GB |
| 显存占用 | 8 GB | 2 GB |
| 推理速度 | 20 tok/s | 50 tok/s |
| 精度损失 | - | < 1% |

---

## 推理部署

### vLLM（推荐）

```bash
vllm serve /path/to/model-awq \
    --quantization awq \
    --dtype half \
    --max-model-len 8192 \
    --port 7890
```

### transformers

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Qwen3-VL 加载
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/path/to/model-awq",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained("/path/to/model-awq", trust_remote_code=True)
```

---

## 常见问题

**Q: Qwen3-VL 应该用 AWQ 还是 GPTQ？**
A: 推荐 GPTQ。GPTQModel 原生支持 Qwen3-VL，同时量化 LLM + ViT。AWQ 仅量化 LLM。

**Q: AutoAWQ 支持 Qwen3-VL 吗？**
A: 不支持。AutoAWQ 仅支持 Qwen2-VL/Qwen2.5-VL，且已被弃用。

**Q: 为什么 AutoAWQ 不支持 Qwen3-VL？**
A: Qwen3-VL 有 DeepStack 特性（融合 ViT 第 5、11、17 层特征），架构与 Qwen2-VL 不同。AutoAWQ 已停止维护，不会添加新模型支持。

**Q: 如何加载 AWQ 量化后的 Qwen3-VL？**
A: 使用专用类：
```python
from transformers import Qwen3VLForConditionalGeneration
model = Qwen3VLForConditionalGeneration.from_pretrained("/path/to/model-awq")
```

---

## 相关链接

- [Qwen3-VL-4B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct)
- [AutoAWQ（已弃用）](https://github.com/casper-hansen/AutoAWQ)
- [llmcompressor（推荐）](https://github.com/vllm-project/llm-compressor)
- [GPTQModel（推荐 Qwen3-VL）](https://github.com/ModelCloud/GPTQModel)
