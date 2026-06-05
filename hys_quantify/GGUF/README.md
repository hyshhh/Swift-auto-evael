# GGUF 量化

> GGUF 是 Hugging Face Hub 和 llama.cpp 生态常用的本地推理模型格式，用一个二进制文件保存权重张量和模型元数据。

## 一句话总结

> **GGUF 适合 llama.cpp、Ollama、LM Studio 等本地推理生态；是否能转换 Qwen3-VL 取决于当前 llama.cpp 对该多模态架构的支持。**

---

## 参考文档

- Hugging Face Hub GGUF 文档：https://huggingface.co/docs/hub/en/gguf
- llama.cpp 仓库：https://github.com/ggml-org/llama.cpp

---

## 快速开始

```bash
# 1. 环境准备
conda create -n gguf python=3.10 -y
conda activate gguf
pip install -U huggingface_hub transformers sentencepiece protobuf numpy

# 2. 量化
bash run.sh
```

`run.sh` 默认会：

1. 克隆并构建 `llama.cpp`
2. 将 Hugging Face 模型转换成 `F16` 的 `.gguf`
3. 再量化成 `Q4_K_M` 的 `.gguf`

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `run.sh` | 一键转换并量化为 GGUF |
| `README.md` | GGUF 使用说明 |

---

## 默认路径

| 变量 | 默认值 |
|------|--------|
| `MODEL` | `/media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-4B-Instruct` |
| `OUTPUT_DIR` | `/media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-4B-GGUF` |
| `LLAMA_CPP_DIR` | `/media/ddc/新加卷/hys/hysnew3/llama.cpp` |
| `OUTTYPE` | `f16` |
| `QUANT_TYPE` | `Q4_K_M` |

你可以通过环境变量覆盖：

```bash
MODEL=/path/to/model \
OUTPUT_DIR=/path/to/output \
QUANT_TYPE=Q8_0 \
bash run.sh
```

---

## 常见量化类型

| 类型 | 说明 |
|------|------|
| `Q4_K_M` | 体积和质量较均衡，常用默认选择 |
| `Q5_K_M` | 质量更好，体积更大 |
| `Q8_0` | 接近原始精度，体积最大 |

---

## 注意事项

1. `GGUF` 是模型存储和本地推理格式，不是 vLLM 的常规部署格式。
2. Qwen3-VL 是多模态模型，如果当前 `llama.cpp` 转换脚本不支持该结构，脚本会在转换阶段失败；这种情况下继续使用 `GPTQ/AWQ/BnB` 更稳。
3. 多模态模型除了语言权重，还可能需要视觉塔、投影层、处理器配置等额外文件；即使转换成功，也要用目标推理框架做一次图文输入验证。

