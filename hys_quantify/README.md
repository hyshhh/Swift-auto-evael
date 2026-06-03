# 量化工具集

> 支持 BnB NF4、GPTQ、AWQ 三种量化方法，适配 Qwen3-VL 多模态模型

---

## 一、环境搭建

```bash
# BnB 量化环境
conda create -n bnb python=3.10 -y
conda activate bnb
pip install bitsandbytes accelerate peft

# GPTQ 量化环境（推荐）
conda create -n swifthys python=3.10 -y
conda activate swifthys
pip install ms-swift gptqmodel optimum accelerate

# AWQ 量化环境
conda create -n llmpress python=3.10 -y
conda activate llmpress
pip install llmcompressor ms-swift
```

---

## 二、量化命令

### BnB NF4 量化

```bash
conda activate bnb
cd BnB
bash run.sh
```

### GPTQ 量化（推荐）

```bash
conda activate swifthys
cd GPTQ
bash run_gptq.sh
```

### AWQ 量化

```bash
conda activate llmpress
cd AWQ
bash run.sh
```

---

## 三、vLLM 部署命令

### BnB NF4 模型

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve /media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-4B-BnB-NF4 \
    --api-key abc123 \
    --served-model-name Qwen/Qwen3-VL-4B-AWQ \
    --max-model-len 10240 \
    --port 7890 \
    --gpu-memory-utilization 0.15 \
    --max-num-seqs 10 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
```

### GPTQ 模型

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve /media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-4B-GPTQ \
    --api-key abc123 \
    --served-model-name Qwen/Qwen3-VL-4B \
    --max-model-len 10240 \
    --port 7890 \
    --gpu-memory-utilization 0.3 \
    --max-num-seqs 10 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --quantization gptq
```

### AWQ 模型

```bash
vllm serve /media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-4B-AWQ \
    --api-key abc123 \
    --served-model-name Qwen/Qwen3-VL-4B-AWQ \
    --max-model-len 8192 \
    --port 7890 \
    --quantization awq
```

---

## 四、量化参数速查

### BnB 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | （必填） | 原始模型路径 |
| `--output` | （必填） | 输出模型路径 |
| `--bits` | 4 | 量化位数 (4=NF4, 8=INT8) |
| `--double_quant` | False | 双重量化（额外节省 ~5%） |
| `--compute_dtype` | bfloat16 | 计算精度 |
| `--gpu` | None | 指定 GPU（多卡环境必须指定） |
| `--copy_config` | False | 从官方模型复制配置文件 |
| `--official_model` | None | 官方模型路径 |

### GPTQ 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | （必填） | 原始模型路径 |
| `--output` | （必填） | 输出模型路径 |
| `--bits` | 4 | 量化位数 (2/3/4/5/6/8) |
| `--group_size` | 128 | 分组大小 |
| `--sym` | True | 对称量化 |
| `--desc_act` | True | 激活排序（提升精度） |
| `--true_sequential` | True | 真正顺序量化 |
| `--damp_percent` | 0.01 | Hessian 阻尼系数 |
| `--batch_size` | 1 | 量化批次大小 |
| `--lm_head` | False | 是否量化 lm_head |

### AWQ 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | （必填） | 原始模型路径 |
| `--output` | （必填） | 输出模型路径 |
| `--bits` | 4 | 量化位数 (3/4) |
| `--group_size` | 128 | 分组大小 |
| `--dataset` | （必填） | 校准数据集 JSONL 路径 |
| `--max_seq_length` | 2048 | 最大序列长度 |
| `--num_calibration_samples` | 128 | 校准样本数量 |

---

## 五、三种方法对比

| 特性 | BnB NF4 | GPTQ INT4 | AWQ INT4 |
|------|---------|-----------|----------|
| **易用性** | ⭐⭐⭐ 极简 | ⭐⭐ 中等 | ⭐⭐ 中等 |
| **量化速度** | ⚡ 极快（分钟） | 慢（小时） | 中等 |
| **推理速度** | 慢（~10 tok/s） | ⚡ 快（~40 tok/s） | ⚡ 快（~35 tok/s） |
| **显存效率** | 中等 | ⚡ 高 | ⚡ 高 |
| **精度** | 高 | 中 | 中 |
| **多模态** | ⚡ 原生支持 | ⚡ 原生支持 | ⚠ 需 patch |
| **QLoRA** | ⚡ 原生支持 | 需配置 | 需配置 |
| **vLLM 部署** | ⚠ 慢（需装 bnb） | ✅ 快 | ✅ 快 |
| **校准数据** | 不需要 | 需要 | 需要 |

### 适用场景

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 快速实验 | **BnB** | 不需要校准数据，分钟级完成 |
| QLoRA 微调 | **BnB** | 原生支持，显存节省 60%+ |
| 生产部署 | **GPTQ** | 推理快，vLLM 原生支持 |
| 多模态推理 | **GPTQ** | 同时量化 LLM + ViT |
| 显存受限 | **GPTQ/AWQ** | 压缩率 75%，显存占用最低 |

---

## 六、量化格式详解

### 4-bit 格式对比

| 格式 | 类型 | 级别数 | 分布 | 用在哪 |
|------|------|--------|------|--------|
| INT4 | 整数 | 16 | 均匀 | GPTQ / AWQ |
| FP4 | 浮点 (E2M1) | 16 | 均匀（有指数位） | bitsandbytes（不推荐） |
| NF4 | 正态浮点 | 16 | 非均匀（匹配正态分布） | bitsandbytes（推荐） |

### 8-bit 格式对比

| 格式 | 结构 | 尾数精度 | 范围 |
|------|------|----------|------|
| FP8 E4M3 | 1符号+4指数+3尾数 | 高（8级/指数） | 中 |
| FP8 E5M2 | 1符号+5指数+2尾数 | 中（4级/指数） | 大 |
| INT8 | 整数 | 256级均匀 | 通用 |

### 量化原理

**BnB NF4**（动态量化）：
```
权重 → 16个预设的非均匀级别（基于正态分布）→ 4-bit 存储
推理时：实时反量化 → 计算（慢）
```

**GPTQ INT4**（静态量化）：
```
权重 → 校准数据优化 → 逐层最小化重建误差 → 4-bit 存储
推理时：预计算好的量化参数 → 直接计算（快）
```

**AWQ INT4**（静态量化）：
```
权重 → 校准数据优化 → 保护重要权重通道 → 4-bit 存储
推理时：预计算好的量化参数 → 直接计算（快）
```

### 为什么 BnB 推理慢？

```
BnB: 每次前向传播 → 反量化 4-bit→16-bit → 计算 → 结果
GPTQ/AWQ: 每次前向传播 → 直接用预计算参数计算 → 结果
                ↑
        BnB 多了反量化步骤，开销约 20-40%
```

---

## 七、模型架构

### Qwen3-VL 结构

```
Qwen3VLForConditionalGeneration
├── model (Qwen3VLModel)
│   ├── embed_tokens
│   ├── layers × N
│   ├── norm
│   └── visual (Qwen3VisionTransformer) ← 视觉编码器
└── lm_head
```

### 各方法的量化策略

| 方法 | LLM 部分 | 视觉编码器 |
|------|----------|------------|
| BnB | NF4 量化 | FP16 保持 |
| GPTQ | GPTQ 量化 | GPTQ 量化（同时量化） |
| AWQ | AWQ 量化 | FP16 保持（不量化） |

---

## 八、性能指标（Qwen3-VL-4B）

| 指标 | FP16 | BnB NF4 | GPTQ INT4 | AWQ INT4 |
|------|------|---------|-----------|----------|
| 模型大小 | ~8 GB | ~2.5 GB | ~1 GB | ~1 GB |
| 显存占用 | ~16 GB | ~4 GB | ~2 GB | ~2 GB |
| 推理速度 | ~20 tok/s | ~10 tok/s | ~40 tok/s | ~35 tok/s |
| 精度损失 | - | < 1% | < 2% | < 1% |

---

## 九、文件结构

```
hys_quantify/
├── README.md                # 本文档
├── BnB/                     # BnB NF4 量化
│   ├── quantize_bnb.py      # 量化脚本
│   ├── verify_bnb.py        # 验证脚本
│   ├── run.sh               # 启动脚本
│   └── config_bnb_qlora_sft.json
├── GPTQ/                    # GPTQ 量化
│   ├── quantize_gptq.py     # 量化脚本
│   ├── run_gptq_swift.py    # ms-swift 量化
│   └── run_gptq.sh          # 启动脚本
└── AWQ/                     # AWQ 量化
    ├── quantize_awq.py      # llmcompressor 量化
    ├── quantize_autoawq.py  # AutoAWQ 量化（实验性）
    ├── run.sh               # 启动脚本
    └── run_llmcompressor.sh
```

---

## 十、常见问题

**Q: vLLM 报错 `No module named 'bitsandbytes'`？**
A: BnB 模型需要安装 bitsandbytes：`pip install bitsandbytes>=0.48.1`

**Q: vLLM 报错 `FlashAttention only supports Ampere GPUs or newer`？**
A: RTX 2080 Ti 不支持 FlashAttention，用 A6000：`CUDA_VISIBLE_DEVICES=1`

**Q: BnB 量化保存卡住？**
A: `save_pretrained()` 和 safetensors 对 BnB 参数有兼容性问题，脚本已用 `torch.save` 绕过。

**Q: vLLM 加载 BnB 模型很慢？**
A: BnB 是动态量化，每次加载都要实时转换，约 5-15 分钟。GPTQ/AWQ 加载快得多。

**Q: 三种方法怎么选？**
A: 快速实验/QLoRA → BnB，生产部署 → GPTQ，传统模型 → AWQ。

---

## 相关链接

- [vLLM INT4 量化文档](https://docs.vllm.com.cn/en/latest/features/quantization/int4/)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [GPTQModel](https://github.com/ModelCloud/GPTQModel)
- [llmcompressor](https://github.com/vllm-project/llm-compressor)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
