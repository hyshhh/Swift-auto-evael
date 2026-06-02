<div align="center">

# 🚀 Swift-auto-evael

### Qwen3.5 模型微调与部署解决方案

解决 ms-swift 框架合并 LoRA 权重后与 vLLM 推理引擎的兼容性问题

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

---

## 📋 目录

- [项目背景](#-项目背景)
- [快速开始](#-快速开始)
- [完整流程](#-完整流程)
- [环境要求](#-环境要求)
- [常见问题](#-常见问题)
- [许可证](#-许可证)

---

## 🎯 项目背景
### 环境配置

    conda create -n swifthys python=3.10 -y
    conda activate swifthys
    pip install ms-swift -U
    pip install "qwen_vl_utils>=0.0.14" decord
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

### 下载模型

```bash
# 下载 Qwen3-VL-4B 到本地
modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir /media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-4B-Instruct

# 下载 Qwen3.5-2B 到本地
modelscope download --model Qwen/Qwen3.5-2B --local_dir /media/ddc/新加卷/hys/hysnew3/model/Qwen3.5-2B

# 下载 Qwen3.5-VL-2B 到本地
modelscope download --model Qwen/Qwen3.5-VL-2B-Instruct --local_dir /media/ddc/新加卷/hys/hysnew3/model/Qwen3.5-VL-2B-Instruct
```

### 问题描述

在使用 ms-swift 框架对 Qwen3.5 模型进行 LoRA 微调后，通过 `swift export --merge_lora` 命令合并权重时，会遇到以下问题：

| 问题 | 影响 |
|------|------|
| 🔴 **配置文件缺失** | 合并后的模型缺少 `video_preprocessor_config.json` 等关键配置文件 |
| 🔴 **vLLM 不兼容** | vLLM 对 Qwen3.5 模型架构的支持不完整，导致推理失败 |
| 🔴 **参数加载异常** | 部分层的权重参数无法正确加载 |

### 解决方案

本项目提供了一套完整的解决方案：

| 步骤 | 说明 | 命令 |
|------|------|------|
| ✅ **合并 LoRA** | 使用 `swift export` 合并权重 | `swift export --merge_lora true` |
| ✅ **修复配置** | 复制缺失的配置文件 | `cp config.json ...` |
| ✅ **VLLM 推理** | 正确配置 vLLM 服务参数 | `vllm serve ...` |
| ✅ **评测脚本** | 基于 vLLM API 的行为识别评测 | `python eval_behavior.py` |

---

## 🚀 快速开始

### 1. 基础推理（LoRA 动态叠加）

```bash
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
    --adapters /media/ddc/新加卷/hys/qmy/ms-swift/output/v2-20260528-211924/checkpoint-31 \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
```

### 2. 合并 LoRA 权重

```bash
CUDA_VISIBLE_DEVICES=2 \
swift export \
    --model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
    --adapters /media/ddc/新加卷/hys/qmy/ms-swift/output/v2-20260528-211924/checkpoint-31 \
    --merge_lora true \
    --output_dir /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b
```

### 3. 修复合并后的模型配置

```bash
# 复制官方模型的配置文件
cp /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B/config.json /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/
cp /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B/video_preprocessor_config.json /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/

# 复制 tokenizer 相关文件（可选，但推荐）
cp /media/ddc/新加卷/hys/hysnew/Qwen3.5-2B-AWQ/tokenizer_config.json /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/
cp /media/ddc/新加卷/hys/hysnew/Qwen3.5-2B-AWQ/vocab.json /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/
cp /media/ddc/新加卷/hys/hysnew/Qwen3.5-2B-AWQ/merges.txt /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/

# 删除多余的文件
rm /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/processor_config.json
rm /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/args.json
```

### 4. 使用 VLLM 推理

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b \
    --api-key abc123 \
    --served-model-name Qwen/Qwen3-VL-4B-AWQ \
    --max-model-len 10240 \
    --port 7890 \
    --gpu-memory-utilization 0.15 \
    --max-num-seqs 10 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
```

### 5. 评测

```bash
python eval_behavior.py \
    --vllm_url http://localhost:7890 \
    --model_name Qwen/Qwen3-VL-4B-AWQ \
    --dataset /media/ddc/新加卷/hys/qmy/agent/data/sft_val.jsonl \
    --debug
```

---

## 📚 完整流程

### SFT 训练

```bash
conda activate swift

CUDA_VISIBLE_DEVICES=1 \
swift sft \
    --model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
    --tuner_type lora \
    --dataset /media/ddc/新加卷/hys/qmy/agent/data/sft_train.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 2 \
    --lora_alpha 8 \
    --target_modules all-linear \
    --gradient_accumulation_steps 32 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 6 \
    --model_author swift \
    --model_name swift-robot
```

### BnB NF4 量化（最简单，支持 vLLM 部署）

```bash
# 激活环境
conda activate bnb

# 安装依赖
pip install bitsandbytes accelerate peft

# 运行量化
cd hys_quantify/BnB
bash run.sh

# 验证量化模型
python verify_bnb.py --model /media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-4B-BnB-NF4 --bits 4

# 使用 vLLM 部署
vllm serve /media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-4B-BnB-NF4 \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port 7890 \
    --api-key abc123 \
    --served-model-name Qwen/Qwen3-VL-4B-BnB
```

### GPTQ 量化

```bash
# 激活环境
conda activate swifthys

# 安装依赖
pip install gptqmodel optimum accelerate

# 运行量化
cd hys_quantify/GPTQ
bash run_gptq.sh
```

### AWQ 量化

```bash
# 激活环境
conda activate llmpress

# 安装依赖
pip install llmcompressor autoawq

# 运行量化
cd hys_quantify/AWQ
bash run.sh
```

---

## 🛠️ 环境要求

| 组件 | 版本要求 | 推荐版本 | 说明 |
|------|----------|----------|------|
| Python | >=3.10 | 3.10 | 运行环境 |
| CUDA | >=12.0 | 12.4 | GPU 支持 |
| PyTorch | >=2.0 | 2.6.0 | 深度学习框架 |
| transformers | >=5.0.0.dev | - | 模型加载 |
| ms-swift | >=4.2.0 | 4.2.2 | 训练框架 |

### 安装依赖

```bash
# 基础环境
conda create -n swifthys python=3.10 -y
conda activate swifthys

# 安装 ms-swift
pip install ms-swift -U

# 安装多模态依赖
pip install "qwen_vl_utils>=0.0.14" decord

# 安装 PyTorch（根据你的 CUDA 版本）
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

---

## ❓ 常见问题

### Q: 合并后模型无法推理？

**A:** 需要复制配置文件：

```bash
cp /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B/config.json /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/
cp /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B/video_preprocessor_config.json /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b/
```

### Q: vLLM 报错不支持 Qwen3.5？

**A:** 确保使用最新版本的 vLLM：

```bash
pip install vllm --upgrade
```

### Q: 显存不足？

**A:** 尝试使用量化或减小 batch size：

```bash
# 使用 BnB NF4 量化（最简单）
cd hys_quantify/BnB && bash run.sh

# 使用 vLLM 部署 BnB 量化模型
vllm serve /path/to/model-bnb \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --dtype bfloat16

# 减小 batch size
--per_device_train_batch_size 1
```

### Q: 如何使用自定义数据集？

**A:** 准备 JSONL 格式的数据集：

```json
{
    "query": "请识别视频中的行为",
    "response": "{\"behavior_id\": \"1\", \"description\": \"...\"}",
    "images": ["/path/to/image1.jpg"]
}
```

---

## 📊 训练指标说明

| 指标 | 说明 | 正常范围 |
|------|------|----------|
| `loss` | 损失值，越低越好 | 0.1 - 1.0 |
| `grad_norm` | 梯度范数，监控梯度爆炸/消失 | 0.1 - 10.0 |
| `learning_rate` | 当前学习率 | 1e-5 - 1e-4 |
| `token_acc` | token 准确率 | 0.8 - 1.0 |
| `memory(GiB)` | 显存占用 | 根据 GPU 而定 |

### 总 Step 数计算公式

```
总 step 数 = 数据量 ÷ (per_device_train_batch_size × gradient_accumulation_steps × GPU数) × epochs
```

---

## 📁 项目结构

```
Swift-auto-evael/
├── README.md                    # 项目说明文档
├── eval_behavior.py             # 行为识别评测脚本
└── hys_quantify/                # 量化脚本集合
    ├── README-量化方法.md        # 量化方法总览
    ├── BnB/                     # BnB NF4 量化（最简单，支持 vLLM）
    │   ├── quantize_bnb.py      # 量化主脚本
    │   ├── verify_bnb.py        # 量化验证脚本
    │   ├── run.sh               # 一键启动脚本
    │   └── README.md            # 说明文档
    ├── GPTQ/                    # GPTQ 量化（推荐生产部署）
    │   ├── quantize_gptq.py     # 量化主脚本
    │   ├── run_gptq.sh          # 一键启动脚本
    │   └── README.md            # 说明文档
    └── AWQ/                     # AWQ 量化
        ├── quantize_awq.py      # 量化主脚本
        ├── run.sh               # 一键启动脚本
        └── README.md            # 说明文档
```

---

## 📄 许可证

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 📚 引用

If you find this project useful, please cite:

```bibtex
@article{swift2024,
  title={SWIFT: Scalable lightWeight Infrastructure for Fine-Tuning},
  author={ModelScope Team},
  journal={arXiv preprint arXiv:2408.05517},
  year={2024}
}
```

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个 Star ⭐**

</div>
