# AWQ 量化

> Activation-aware Weight Quantization，激活感知权重量化

## 一句话总结

> **AWQ 通过保护重要权重通道，实现高精度 4-bit 量化，适合生产部署。**

---

## 快速开始

```bash
# 1. 环境准备
conda create -n awq python=3.10 -y
conda activate awq
pip install llmcompressor ms-swift

# 2. 量化
bash run.sh

# 3. 验证
python verify.py --model /path/to/output
```

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `quantize_awq.py` | 使用 llmcompressor 量化（推荐） |
| `quantize_autoawq.py` | 使用 AutoAWQ 量化 |
| `run.sh` | 一键启动脚本 |
| `run_llmcompressor.sh` | llmcompressor 直接启动脚本 |

---

## 量化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | （必填） | 原始模型路径 |
| `--output` | （必填） | 输出模型路径 |
| `--bits` | 4 | 量化位数 |
| `--group_size` | 128 | 分组大小 |
| `--dataset` | alpaca-en | 校准数据集 |

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

### vLLM

```bash
vllm serve /path/to/model-awq \
    --quantization awq \
    --max-model-len 8192 \
    --port 8000
```

### transformers

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/model-awq",
    device_map="auto",
    trust_remote_code=True,
)
```

---

## 常见问题

**Q: 报错 `qwen3_5 isn't supported yet`**
A: 升级 autoawq: `pip install autoawq>=0.2.9`

**Q: 显存不足**
A: 减小 `--group_size` 或减少校准数据

---

## 相关链接

- [AWQ 论文](https://arxiv.org/abs/2306.00978)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [llmcompressor](https://github.com/neuralmagic/llmcompressor)
