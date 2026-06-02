# AWQ 量化

> Activation-aware Weight Quantization，激活感知权重量化

## 一句话总结

> **使用 llmcompressor 进行 AWQ 量化，视觉编码器保持 FP16，仅量化语言模型部分。**

---

## 参考实现

本方案参考 [Junfeng-Pan/Qwen3-AWQ](https://github.com/Junfeng-Pan/Qwen3-AWQ) 实现，使用 `llmcompressor` 进行 Qwen3-VL 多模态模型的 AWQ 量化。

**量化效果（Qwen3-VL-8B-Instruct）：**
- 模型大小：~16.5 GB → **6.8 GB**（减少 59%）
- 显存占用：>16 GB → **~7.5 GB**
- OCRBench 准确率：**87.4%**（保持良好）

---

## 快速开始

```bash
# 1. 环境准备
conda create -n awq python=3.10 -y
conda activate awq
pip install llmcompressor==0.9.0 transformers>=4.57.0

# 2. 量化
bash run.sh

# 3. 验证
python verify.py --model /path/to/output
```

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `quantize_awq.py` | llmcompressor 量化脚本（推荐） |
| `quantize_autoawq.py` | AutoAWQ 量化脚本（实验性，不推荐） |
| `run.sh` | 一键启动脚本 |
| `run_llmcompressor.sh` | llmcompressor 直接启动脚本 |

---

## 量化原理

### Qwen3-VL 模型结构

```
Qwen3-VL-4B-Instruct
├── 语言模型 (LLM)      → AWQ 4-bit 量化 ✅
├── 视觉编码器 (ViT)    → FP16 保持（不量化）⚠️
└── Projection 层       → AWQ 4-bit 量化 ✅
```

### 关键设计

1. **视觉编码器保护**：通过 `ignore=["re:model[.]visual.*"]` 将视觉塔排除在量化之外
2. **手动层级映射**：Qwen3-VL 有 `q_norm`/`k_norm` 等特殊层，需要手动配置 `AWQMapping`
3. **Processor 代替 Tokenizer**：多模态模型需要使用 `AutoProcessor`

### AWQ 层级映射配置

```python
mappings = [
    AWQMapping(
        smooth_layer="re:.*input_layernorm",
        balance_layers=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
    ),
    AWQMapping(
        smooth_layer="re:.*post_attention_layernorm",
        balance_layers=["re:.*gate_proj", "re:.*up_proj"],
    ),
    AWQMapping(
        smooth_layer="re:.*up_proj",
        balance_layers=["re:.*down_proj"],
    ),
]

ignore = [
    "re:.*embed_tokens",
    "re:model[.]visual.*",  # 忽略视觉塔
    "re:visual.*",
    "lm_head",
    "re:.*q_norm",  # 忽略 attention 内部的 norm
    "re:.*k_norm",
]
```

---

## 量化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | （必填） | 原始模型路径 |
| `--output` | （必填） | 输出模型路径 |
| `--bits` | 4 | 量化位数（3/4） |
| `--group_size` | 128 | 分组大小 |
| `--dataset` | （必填） | 校准数据集 JSONL 路径 |
| `--max_seq_length` | 2048 | 最大序列长度 |
| `--num_calibration_samples` | 128 | 校准样本数量 |

---

## 使用方法

### 方法 1：使用运行脚本（推荐）

```bash
cd hys_quantify/AWQ
bash run.sh
```

### 方法 2：直接运行

```bash
CUDA_VISIBLE_DEVICES=0 python quantize_awq.py \
    --model /path/to/Qwen3-VL-4B-Instruct \
    --output /path/to/Qwen3-VL-4B-AWQ \
    --bits 4 \
    --group_size 128 \
    --max_seq_length 2048 \
    --num_calibration_samples 128 \
    --dataset /path/to/calibration.jsonl \
    --copy_config \
    --official_model /path/to/Qwen3-VL-4B-Instruct
```

---

## 校准数据集格式

JSONL 格式，支持多种字段名：

```json
{"query": "描述这张图片", "response": "这是一张..."}
{"messages": [{"role": "user", "content": "描述这张图片"}, {"role": "assistant", "content": "这是一张..."}]}
{"text": "一段文本内容"}
```

---

## 依赖项

```
llmcompressor==0.9.0
transformers>=4.57.0
accelerate>=1.12.0
torch>=2.7.0
```

---

## 推理部署

### vLLM（推荐）

```bash
vllm serve /path/to/Qwen3-VL-4B-AWQ \
    --quantization awq \
    --dtype half \
    --max-model-len 8192 \
    --port 7890 \
    --api-key abc123
```

### transformers

```python
from transformers import AutoModelForVision2Seq, AutoProcessor

model = AutoModelForVision2Seq.from_pretrained(
    "/path/to/Qwen3-VL-4B-AWQ",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained("/path/to/Qwen3-VL-4B-AWQ", trust_remote_code=True)
```

---

## 常见问题

**Q: 为什么视觉编码器不量化？**
A: 视觉编码器对量化敏感，且参数量占比相对较小。保持 FP16 可以保护多模态理解能力。

**Q: 为什么需要手动配置 AWQMapping？**
A: Qwen3-VL 是较新架构，自动推断层级关系会失败。且 Qwen3-VL 有 `q_norm`/`k_norm` 等特殊层。

**Q: 与 GPTQ 相比有什么优劣？**
A: AWQ 仅量化 LLM，GPTQ 同时量化 LLM + ViT。如需完整多模态量化，推荐 GPTQ。

**Q: 显存不足怎么办？**
A: 减少 `--num_calibration_samples` 或 `--max_seq_length`

---

## 相关链接

- [Junfeng-Pan/Qwen3-AWQ（参考实现）](https://github.com/Junfeng-Pan/Qwen3-AWQ)
- [llmcompressor](https://github.com/vllm-project/llm-compressor)
- [Qwen3-VL-4B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct)
- [vLLM 文档](https://docs.vllm.ai/)
