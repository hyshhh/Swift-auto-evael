# GPTQ 量化

> GPT-Optimized Quantization，基于 OBS 的逐层量化

## 一句话总结

> **GPTQ 是最成熟的 4-bit 量化方案，推理速度快，支持多模态，推荐生产部署。**

---

## 快速开始

```bash
# 1. 环境准备
conda create -n gptq python=3.10 -y
conda activate gptq
pip install ms-swift gptqmodel optimum accelerate

# 2. 量化
bash run.sh

# 3. 验证
python verify.py --model /path/to/output
```

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `quantize_gptq.py` | GPTQ 量化脚本 |
| `run_gptq_swift.py` | 使用 ms-swift 量化（推荐） |
| `run.sh` | 一键启动脚本 |
| `run_gptq.sh` | GPTQ 直接启动脚本 |

---

## 量化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | （必填） | 原始模型路径 |
| `--output` | （必填） | 输出模型路径 |
| `--bits` | 4 | 量化位数 (2/3/4/5/6/8) |
| `--group_size` | 128 | 分组大小 |
| `--sym` | True | 对称量化 |
| `--desc_act` | True | 激活排序 |

---

## 性能指标

| 指标 | FP16 | GPTQ 4-bit |
|------|------|------------|
| 模型大小 | 4 GB | 1 GB |
| 显存占用 | 8 GB | 1.6 GB |
| 推理速度 | 20 tok/s | 47 tok/s |
| 精度损失 | - | < 2% |

---

## 推理部署

### vLLM

```bash
vllm serve /path/to/model-gptq \
    --quantization gptq \
    --dtype float16 \
    --max-model-len 8192 \
    --port 8000
```

### LMDeploy

```bash
lmdeploy serve api_server \
    /path/to/model-gptq \
    --backend turbomind \
    --model-format gptq \
    --server-port 8000
```

### transformers

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/model-gptq",
    device_map="auto",
    trust_remote_code=True,
)
```

---

## 多模态支持

GPTQ 对 Qwen3.5 多模态架构支持最好：

```
量化策略：
├── 语言模型 (LLM)  → GPTQ 4-bit 量化
└── 视觉编码器 (VE) → GPTQ 4-bit 量化（同时量化）

结果：完整的多模态推理 ✓
```

---

## 常见问题

**Q: 报错 `权重名称前缀不匹配`**
A: 使用 `--copy_config --official_model` 参数

**Q: vLLM 加载失败**
A: 尝试 LMDeploy 或 transformers 直接加载

**Q: 显存不足**
A: 增大 `--group_size` 或减小 `--batch_size`

---

## 相关链接

- [GPTQ 论文](https://arxiv.org/abs/2210.17323)
- [GPTQModel](https://github.com/ModelCloud/GPTQModel)
- [ms-swift](https://github.com/modelscope/ms-swift)
