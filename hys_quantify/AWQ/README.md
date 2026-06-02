# AWQ 量化

> Activation-aware Weight Quantization，激活感知权重量化

## 一句话总结

> **AWQ 通过保护重要权重通道，实现高精度 4-bit 量化，vLLM 原生支持，适合生产部署。**

---

## 快速开始

```bash
# 1. 环境准备
conda create -n awq python=3.10 -y
conda activate awq
pip install autoawq ms-swift

# 2. 量化
bash run.sh

# 3. 验证
python verify.py --model /path/to/output
```

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `quantize_autoawq.py` | 使用 AutoAWQ 量化（推荐，vLLM 原生支持） |
| `quantize_awq.py` | 使用 llmcompressor 量化（支持多模态） |
| `run.sh` | 一键启动脚本（支持选择量化方式） |
| `run_llmcompressor.sh` | llmcompressor 直接启动脚本 |

---

## 量化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | （必填） | 原始模型路径 |
| `--output` | （必填） | 输出模型路径 |
| `--bits` | 4 | 量化位数（3/4） |
| `--group_size` | 128 | 分组大小 |
| `--dataset` | None | 校准数据集路径 |
| `--version` | GEMM | 量化内核版本（GEMM/GEMV/Marlin） |

---

## 多模态支持（Qwen3-VL）

AWQ 支持 Qwen3-VL 多模态模型量化：

```
Qwen3-VL 结构：
├── 语言模型 (LLM)      → AWQ 4-bit 量化
├── 视觉编码器 (ViT)    → FP16 保持（llmcompressor）或 AWQ 量化（AutoAWQ）
└── Projection 层       → AWQ 4-bit 量化
```

### 两种量化方式对比

| 特性 | AutoAWQ | llmcompressor |
|------|---------|---------------|
| 推荐度 | ⭐⭐⭐ 推荐 | ⭐⭐ 备选 |
| vLLM 兼容 | ✅ 原生支持 | ✅ 支持 |
| 多模态支持 | ✅ 支持 | ✅ 支持 |
| 量化速度 | ⚡ 快 | 中等 |
| 依赖 | `autoawq` | `llmcompressor` |

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

**Q: 报错 `qwen3_5 isn't supported yet`**
A: 升级 autoawq: `pip install autoawq>=0.2.9`

**Q: 报错 `No module named 'awq'`**
A: 安装 autoawq: `pip install autoawq`

**Q: 显存不足**
A: 减小 `--group_size`（如 64）或减少校准数据

**Q: 量化后模型无法加载**
A: 检查是否正确复制了配置文件，使用 `--copy_config --official_model` 参数

**Q: vLLM 加载时报错**
A: 确保使用 `--quantization awq` 参数，或检查 config.json 中是否包含 `quantization_config`

---

## 环境依赖

```bash
# 基础依赖
pip install autoawq ms-swift

# 如遇编译问题
pip install autoawq --no-build-isolation

# vLLM 推理
pip install vllm
```

---

## 相关链接

- [AWQ 论文](https://arxiv.org/abs/2306.00978)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [vLLM AutoAWQ 文档](https://docs.vllm.ai/en/latest/features/quantization/auto_awq.html)
- [llmcompressor](https://github.com/neuralmagic/llmcompressor)
