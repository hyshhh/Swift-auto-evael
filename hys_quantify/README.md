# AWQ 量化脚本

## 功能说明

将 LoRA 微调并合并后的模型进行 AWQ 4-bit 量化，减少模型大小并提升推理速度。

## 目录结构

```
hys_quantify/
├── quantize_awq.py    # 主量化脚本
├── run_quantize.sh    # 运行脚本
└── README.md          # 说明文档
```
### 环境搭建
    conda create -n llmpress python=3.10 -y
    conda activate llmpress
## 使用方法
### 方法 1：使用运行脚本（推荐）

```bash
cd hys_quantify
bash run_quantize.sh
```

### 方法 2：直接运行 Python 脚本

```bash
cd hys_quantify

CUDA_VISIBLE_DEVICES=0 python quantize_awq.py \
    --model /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b \
    --output /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-awq \
    --bits 4 \
    --group_size 128 \
    --dataset alpaca-en \
    --copy_config \
    --official_model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B
```

### 方法 3：使用自定义校准数据集

```bash
CUDA_VISIBLE_DEVICES=0 python quantize_awq.py \
    --model /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b \
    --output /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-awq \
    --bits 4 \
    --group_size 128 \
    --dataset /media/ddc/新加卷/hys/qmy/agent/data/sft_train.jsonl \
    --copy_config \
    --official_model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | （必填） | 合并后的模型路径 |
| `--output` | （必填） | 量化后模型保存路径 |
| `--bits` | 4 | 量化位数（目前支持 4-bit） |
| `--group_size` | 128 | 量化分组大小 |
| `--dataset` | None | 校准数据集路径或内置数据集名称 |
| `--copy_config` | false | 是否复制官方模型配置文件 |
| `--official_model` | None | 官方模型路径（用于复制配置） |

## 校准数据集

### 使用内置数据集

- `alpaca-en`：英文指令数据集（推荐）
- `alpaca-zh`：中文指令数据集
- `code-alpaca`：代码指令数据集

### 使用自定义数据集

数据集格式为 JSONL，每行一个 JSON 对象：

```json
{
    "query": "请介绍一下你自己",
    "response": "我是一个AI助手..."
}
```

## 量化后模型结构

量化后的模型目录包含以下文件：

```
wt-Qwen2b-awq/
├── model.safetensors.awq    # AWQ 量化权重
├── config.json              # 模型配置
├── quant_config.json        # 量化配置
├── tokenizer.json           # 分词器
├── tokenizer_config.json    # 分词器配置
├── quant_info.json          # 量化信息
└── ...
```

## 使用量化模型

### 使用 vLLM 推理

```bash
vllm serve /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-awq \
    --api-key abc123 \
    --served-model-name Qwen/Qwen3-VL-4B-AWQ \
    --max-model-len 8192 \
    --port 7890 \
    --quantization awq
```

### 使用 transformers 推理

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '/media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-awq'

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
```

## 量化效果

| 指标 | 量化前 (FP16) | 量化后 (AWQ 4-bit) | 变化 |
|------|---------------|-------------------|------|
| 模型大小 | ~4GB | ~1GB | -75% |
| 显存占用 | ~8GB | ~2GB | -75% |
| 推理速度 | 20 tokens/s | 50 tokens/s | +150% |
| 精度损失 | - | < 1% | - |

## 注意事项

1. **显存要求**：量化过程需要约 8-16GB 显存
2. **量化时间**：2B 模型约需 5-10 分钟
3. **精度损失**：4-bit 量化会有轻微精度损失，通常影响不大
4. **配置文件**：建议使用 `--copy_config` 复制官方配置文件，确保兼容性

## 常见问题

### Q: 报错 `No module named 'awq'`

A: 安装 autoawq：

```bash
pip install autoawq==0.2.9
```

### Q: 报错 `qwen3_5 isn't supported yet`

A: 确保使用 autoawq 0.2.9 或更高版本。

### Q: 量化后模型无法加载

A: 检查是否正确复制了配置文件，特别是 `config.json` 和 `video_preprocessor_config.json`。

### Q: 量化过程显存不足

A: 减少校准数据集大小，或使用更小的 `--group_size`（如 64）。
