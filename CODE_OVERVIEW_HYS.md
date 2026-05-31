# 项目代码整理说明

这份文档用于快速理解当前仓库的结构，以及你在原始 `ms-swift` 框架上新增的训练、量化、评测和兼容性修复流程。

## 1. 项目定位

当前仓库基于 ModelScope 的 `ms-swift`，核心能力是大模型训练、微调、推理、部署、导出、量化和评测。

你当前主要使用它做：

- Qwen3.5 / Qwen VL 类模型的 LoRA SFT 微调
- LoRA 权重合并
- 合并后模型配置修复
- vLLM 推理部署
- 行为识别任务评测
- AWQ 4-bit 量化
- Qwen3.5 与 vLLM / transformers 兼容性修复

## 2. 目录结构

```text
ms-swift/
├── swift/                  # 框架核心代码
│   ├── cli/                # swift 命令行入口
│   ├── arguments/          # 各命令参数定义
│   ├── pipelines/          # 训练、推理、导出、评测主流程
│   ├── model/              # 模型注册、加载、模型适配
│   ├── template/           # 对话模板、多模态输入模板
│   ├── dataset/            # 数据集加载与注册
│   ├── trainers/           # SFT / 常规训练 Trainer
│   ├── rlhf_trainers/      # DPO、GRPO、PPO、KTO 等 RLHF Trainer
│   ├── tuners/             # LoRA / Adapter 等微调模块
│   ├── ui/                 # Web UI 相关代码
│   ├── megatron/           # Megatron 训练支持
│   └── utils/              # 通用工具函数
├── examples/               # 官方示例脚本
├── docs/                   # 官方文档
├── tests/                  # 测试代码
├── requirements/           # 可选依赖分组
├── hys_quantify/           # 你新增的 AWQ 量化流程
├── evelhys/                # 你新增的行为识别评测脚本
├── readme-hys-sft.md       # 你的 SFT/推理/量化笔记
├── patch_vllm_qwen35.py    # vLLM Qwen3.5 config 兼容性补丁
├── test_gpu_speed.py       # PaddleOCR GPU 速度测试脚本
├── setup.py                # Python 包安装与 swift 命令注册
└── README.md               # 当前项目说明
```

## 3. 命令入口

安装后主要通过 `swift` 命令使用。入口定义在 `setup.py`：

```python
entry_points={
    'console_scripts': [
        'swift=swift.cli.main:cli_main',
        'megatron=swift.cli._megatron.main:cli_main'
    ]
}
```

`swift.cli.main` 会把命令路由到不同模块：

```text
swift sft       -> swift.cli.sft
swift infer     -> swift.cli.infer
swift export    -> swift.cli.export
swift deploy    -> swift.cli.deploy
swift eval      -> swift.cli.eval
swift web-ui    -> swift.cli.web_ui
swift rlhf      -> swift.cli.rlhf
swift sample    -> swift.cli.sample
swift app       -> swift.cli.app
```

如果设置了 `NPROC_PER_NODE` 或 `NNODES`，训练/推理相关命令会自动通过 `torch.distributed.run` 启动多卡任务。

## 4. 你的主流程

### 4.1 LoRA 动态推理

用于验证训练出的 adapter 是否有效，不合并权重：

```bash
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
    --adapters /media/ddc/新加卷/hys/qmy/ms-swift/output/v2-20260528-211924/checkpoint-31 \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
```

### 4.2 SFT 训练

笔记文件：`readme-hys-sft.md`

核心参数：

- `--tuner_type lora`：使用 LoRA 微调
- `--quant_bits 4`：4-bit 加载训练
- `--freeze_llm true`、`--freeze_vit true`：冻结大模型和视觉塔
- `--target_modules all-linear`：对线性层挂 LoRA
- `--gradient_accumulation_steps 32`：梯度累积，降低显存压力
- `--max_length 2048`：样本最大长度

### 4.3 合并 LoRA

```bash
CUDA_VISIBLE_DEVICES=2 \
swift export \
    --model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
    --adapters /media/ddc/新加卷/hys/qmy/ms-swift/output/v2-20260528-211924/checkpoint-31 \
    --merge_lora true \
    --output_dir /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b
```

合并后你遇到的关键问题是：导出的模型目录可能缺少 Qwen3.5 / 多模态推理需要的配置文件。

### 4.4 合并后配置修复

需要从官方模型目录复制：

- `config.json`
- `video_preprocessor_config.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`
- `preprocessor_config.json`
- `generation_config.json`

并删除可能造成冲突的：

- `processor_config.json`
- `args.json`

`hys_quantify/quantize_awq.py` 里的 `copy_config_files()` 已经封装了这件事。

### 4.5 vLLM 部署

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

## 5. 量化模块

目录：`hys_quantify/`

主要文件：

- `quantize_awq.py`：使用 `llmcompressor` 做 AWQ 量化
- `quantize_autoawq.py`：AutoAWQ 版本量化脚本
- `run_quantize.sh`：一键运行脚本
- `README.md`：量化说明
- `hys-readme-量化.md`：你的量化笔记

`quantize_awq.py` 的主要逻辑：

1. 解析模型路径、输出路径、量化位数、校准集等参数
2. 设置 `CT_DISABLE_DEVICE_DISPATCH=1` 避免兼容性问题
3. 导入 `swift`，让 transformers 能识别 Qwen3.5 自定义模型
4. 加载模型和 tokenizer
5. 构造 `AWQModifier`
6. 加载本地 JSONL 校准数据或 llmcompressor 注册数据集
7. 执行 `oneshot()`
8. 保存量化模型和 tokenizer
9. 可选复制官方配置文件
10. 写入 `quant_info.json`

常用命令：

```bash
CUDA_VISIBLE_DEVICES=0 python hys_quantify/quantize_awq.py \
    --model /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b \
    --output /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-awq \
    --bits 4 \
    --group_size 128 \
    --dataset /media/ddc/新加卷/hys/qmy/agent/data/sft_val.jsonl \
    --copy_config \
    --official_model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B
```

## 6. 评测模块

目录：`evelhys/`

主要文件：

- `eval_behavior.py`：行为识别评测脚本
- `评测.md`：评测命令笔记

`eval_behavior.py` 的功能：

1. 读取 JSONL 评测集
2. 每条样本取 `query`、`response`、`images`
3. 将本地图片转成 base64，按 OpenAI Chat Completions 多模态格式请求 vLLM
4. 从模型输出中提取 `behavior_id`
5. 和 ground truth 的 `behavior_id` 对比
6. 统计总准确率、按类别准确率、unknown 数量和错误样本
7. 保存 `.eval_result.json`

常用命令：

```bash
python evelhys/eval_behavior.py \
    --vllm_url http://localhost:7890 \
    --model_name Qwen/Qwen3-VL-4B-AWQ \
    --dataset /media/ddc/新加卷/hys/qmy/agent/data/sft_val.jsonl \
    --debug
```

数据格式：

```json
{
  "query": "请识别视频中的行为",
  "response": "{\"behavior_id\": \"1\", \"description\": \"...\"}",
  "images": ["/path/to/image1.jpg"]
}
```

## 7. 兼容性修复

### 7.1 vLLM config 类型冲突

文件：`patch_vllm_qwen35.py`

作用：修改 vLLM 的 `multimodal/processing/context.py`，让 `get_hf_config()` 接受 transformers 和 vLLM 中“类名相同但模块不同”的 Qwen3.5 config 类型。

使用：

```bash
python patch_vllm_qwen35.py
```

注意：这个脚本会直接修改当前 Python 环境里的 vLLM 源码，建议只在固定实验环境中运行。

### 7.2 Qwen3.5 sequence parallel 参数修复

文件：`swift/model/models/qwen.py`

当前本地改动：

```diff
- fwd_kwargs['use_cache'] = kwargs.get('use_cache')
+ fwd_kwargs['use_cache'] = use_cache
```

并且给 `_patch_qwen3_5_linear_attention_sequence_parallel()` 内部 forward 增加了显式参数：

```python
use_cache=None
```

作用：避免 `use_cache` 被吞在 `kwargs` 中导致传参异常。

## 8. 当前工作区状态

当前检测到的本地变更包括：

- `swift/model/models/qwen.py`：Qwen3.5 sequence parallel 的 `use_cache` 修复
- `hys_quantify/README.md`：量化文档修改
- `hys_quantify/hys-readme-量化.md`：量化笔记修改
- `hys_quantify/hys-readme.md`：已删除
- `readme-hys.md`：已删除
- `readme-hys-sft.md`：新增 SFT 笔记
- `.claude/`：新增目录
- `evelhys/`：新增评测目录

另外，`git status` 时出现：

```text
error: could not lock config file C:/Users/86135/.gitconfig: File exists
```

这通常不是项目代码问题，而是 Git 试图写全局配置时锁文件冲突。后续如果 Git 命令继续报这个错，可以检查并处理 `C:/Users/86135/.gitconfig.lock` 或同名异常文件。

## 9. 已做基础检查

以下脚本已通过 Python 语法编译检查：

```text
evelhys/eval_behavior.py
hys_quantify/quantize_awq.py
hys_quantify/quantize_autoawq.py
patch_vllm_qwen35.py
test_gpu_speed.py
```

检查命令：

```bash
python -m py_compile evelhys/eval_behavior.py hys_quantify/quantize_awq.py hys_quantify/quantize_autoawq.py patch_vllm_qwen35.py test_gpu_speed.py
```

## 10. 建议后续整理方向

建议把个人实验相关内容固定成下面这种结构：

```text
hys/
├── sft.md                  # 训练命令和参数说明
├── merge_lora.md           # 合并与配置修复说明
├── deploy_vllm.md          # vLLM 启动命令
├── eval.md                 # 评测命令和指标说明
├── quantize.md             # AWQ 量化说明
└── scripts/
    ├── eval_behavior.py
    ├── quantize_awq.py
    └── patch_vllm_qwen35.py
```

这样可以把官方 `ms-swift` 框架代码和你的实验脚本分开，后续升级官方代码时冲突会少很多。
