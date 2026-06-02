# AWQ 量化

> Activation-aware Weight Quantization，激活感知权重量化

## 一句话总结

> **AWQ 通过保护重要权重通道，实现高精度 4-bit 量化，vLLM 原生支持，适合生产部署。**

---

## ⚠️ 重要说明

### AutoAWQ 已被官方弃用

- **AutoAWQ 已停止维护**，最后测试配置：Torch 2.6.0 + Transformers 4.51.3
- **推荐替代方案**：[llmcompressor](https://github.com/vllm-project/llm-compressor)（vLLM 项目接管）
- **Qwen3-VL 量化**：请使用 `quantize_awq.py`（llmcompressor），不要使用 `quantize_autoawq.py`

### AutoAWQ 官方支持的多模态模型

| 模型 | 状态 |
|------|------|
| Qwen2-VL | ✅ 官方支持 |
| Qwen2.5-VL | ✅ 官方支持 |
| Qwen2.5-Omni | ✅ 官方支持 |
| LLaVA / LLaVA-Next | ✅ 官方支持 |
| Phi3-V | ✅ 官方支持 |
| **Qwen3-VL** | ❌ **不支持**（实验性补丁可用） |

---

## 快速开始

```bash
# 1. 环境准备
conda create -n awq python=3.10 -y
conda activate awq
pip install llmcompressor ms-swift

# 2. 量化（使用 llmcompressor，推荐）
bash run.sh

# 3. 验证
python verify.py --model /path/to/output
```

---

## 文件说明

| 文件 | 用途 | 推荐度 |
|------|------|--------|
| `quantize_awq.py` | llmcompressor 量化（支持 Qwen3-VL） | ⭐⭐⭐ 推荐 |
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

## Qwen3-VL 量化（推荐方式）

使用 llmcompressor 量化 Qwen3-VL：

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
Qwen3-VL 结构：
├── 语言模型 (LLM)      → AWQ 4-bit 量化
├── 视觉编码器 (ViT)    → FP16 保持（llmcompressor 自动处理）
└── Projection 层       → AWQ 4-bit 量化
```

---

## 性能指标

| 指标 | FP16 | AWQ 4-bit |
|------|------|-----------|
| 模型大小 | 8 GB | 2 GB |
| 显存占用 | 16 GB | 4 GB |
| 推理速度 | 20 tok/s | 50 tok/s |
| 精度损失 | - | < 1% |

---

## 推理部署

### vLLM（推荐）

```bash
# 启动 API 服务
vllm serve /path/to/model-awq \
    --api-key abc123 \
    --served-model-name Qwen/Qwen3-VL-4B-AWQ \
    --quantization awq \
    --dtype half \
    --max-model-len 8192 \
    --port 7890
```

```python
# Python API 调用
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/model-awq",
    quantization="awq",
    dtype="half",
    max_model_len=8192,
    gpu_memory_utilization=0.9,
)

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
outputs = llm.generate(["你好，请介绍一下你自己。"], sampling_params)
```

### transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/model-awq",
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("/path/to/model-awq", trust_remote_code=True)
```

---

## 常见问题

**Q: Qwen3-VL 应该用哪个脚本？**
A: 使用 `quantize_awq.py`（llmcompressor），不要使用 `quantize_autoawq.py`

**Q: AutoAWQ 还能用吗？**
A: 对于 Qwen2-VL/Qwen2.5-VL 可以使用，但已停止维护。推荐迁移到 llmcompressor

**Q: 报错 `No module named 'awq'`**
A: 安装 autoawq: `pip install autoawq`

**Q: 显存不足**
A: 减小 `--group_size`（如 64）或减少校准数据

**Q: vLLM 加载时报错**
A: 确保使用 `--quantization awq` 参数

---

## 环境依赖

```bash
# 推荐（llmcompressor）
pip install llmcompressor ms-swift

# 备选（AutoAWQ，仅支持 Qwen2-VL/Qwen2.5-VL）
pip install autoawq ms-swift

# vLLM 推理
pip install vllm
```

---

## 相关链接

- [AWQ 论文](https://arxiv.org/abs/2306.00978)
- [AutoAWQ（已弃用）](https://github.com/casper-hansen/AutoAWQ)
- [llmcompressor（推荐）](https://github.com/vllm-project/llm-compressor)
- [vLLM AutoAWQ 文档](https://docs.vllm.ai/en/latest/features/quantization/auto_awq.html)
