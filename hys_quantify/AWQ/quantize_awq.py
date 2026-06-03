"""
AWQ 量化脚本 - 使用 llm-compressor 进行 AWQ 量化（支持 Qwen3-VL 多模态架构）

用法：
    python quantize_awq.py \
        --model /path/to/merged_model \
        --output /path/to/awq_model \
        --dataset /path/to/calibration.jsonl \
        --bits 4 \
        --gpu 0

说明：
    参考 https://github.com/Junfeng-Pan/Qwen3-AWQ 实现
    支持 Qwen3-VL 多模态模型，视觉编码器保持 FP16 不量化。
    校准数据支持 JSONL 格式。

依赖：
    pip install llmcompressor==0.9.0 transformers>=4.57.0
"""
import os
import json
import shutil
import argparse
from pathlib import Path

# 禁用 compressed_tensors 的设备分发，避免 PyTorch 版本兼容性问题
os.environ['CT_DISABLE_DEVICE_DISPATCH'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='AWQ 量化脚本（使用 llm-compressor）')
    parser.add_argument('--model', type=str, required=True, help='合并后的模型路径')
    parser.add_argument('--output', type=str, required=True, help='量化后模型保存路径')
    parser.add_argument('--bits', type=int, default=4, choices=[2, 3, 4], help='量化位数（默认 4-bit）')
    parser.add_argument('--group_size', type=int, default=128, help='量化分组大小')
    parser.add_argument('--dataset', type=str, default=None, help='校准数据集 JSONL 路径')
    parser.add_argument('--copy_config', action='store_true', help='复制官方模型配置文件')
    parser.add_argument('--official_model', type=str, default=None, help='官方模型路径（用于复制配置）')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='最大序列长度（默认 2048）')
    parser.add_argument('--num_calibration_samples', type=int, default=128, help='校准样本数量')
    parser.add_argument('--gpu', type=str, default=None, help='指定使用的 GPU，如 "0" 或 "0,1"')
    parser.add_argument('--max_shard_size', type=str, default='5GB', help='保存时每个分片的最大大小')
    return parser.parse_args()


def copy_config_files(official_model, output_path):
    """从官方模型复制配置文件"""
    official_model = Path(official_model)
    output_path = Path(output_path)

    config_files = [
        'config.json',
        'video_preprocessor_config.json',
        'tokenizer_config.json',
        'vocab.json',
        'merges.txt',
        'preprocessor_config.json',
        'generation_config.json',
        'processor_config.json',
        'chat_template.json',
        'special_tokens_map.json',
        'tokenizer.json',
        'image_preprocessor_config.json',
    ]

    for config_file in config_files:
        src = official_model / config_file
        dst = output_path / config_file
        if src.exists():
            shutil.copy2(src, dst)
            print(f'复制: {config_file}')

    for extra_file in ['args.json']:
        extra_path = output_path / extra_file
        if extra_path.exists():
            extra_path.unlink()
            print(f'删除: {extra_file}')


def detect_model_type(model_path):
    """检测模型类型"""
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        model_type = config.get('model_type', '')

        if model_type == 'qwen3_vl':
            return 'qwen3_vl'
        elif model_type == 'qwen2_vl':
            return 'qwen2_vl'
        elif 'vision_config' in config or 'visual_config' in config:
            return 'multimodal'

    return 'text_only'


def create_awq_recipe(model_type, bits=4, group_size=128):
    """
    创建 AWQ 量化配方

    参考 https://github.com/Junfeng-Pan/Qwen3-AWQ 的实现
    """
    from llmcompressor.modifiers.awq import AWQModifier, AWQMapping

    if model_type in ('qwen3_vl', 'qwen2_vl', 'multimodal'):
        # 多模态模型的层级映射（手动配置，因为自动推断在 Vision2Seq 模型上会失败）
        mappings = [
            AWQMapping(
                smooth_layer="re:.*input_layernorm",
                balance_layers=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
            ),
            # 注意：不添加 v_proj -> o_proj 映射，因为它们之间隔着 Attention 算子
            # 且 Qwen3-VL 有 q_norm/k_norm
            AWQMapping(
                smooth_layer="re:.*post_attention_layernorm",
                balance_layers=["re:.*gate_proj", "re:.*up_proj"],
            ),
            AWQMapping(
                smooth_layer="re:.*up_proj",
                balance_layers=["re:.*down_proj"],
            ),
        ]

        # 忽略层：视觉编码器、embed_tokens、lm_head、q_norm/k_norm
        ignore = [
            "re:.*embed_tokens",
            "re:model[.]visual.*",  # 忽略视觉塔
            "re:visual.*",
            "lm_head",
            "re:.*q_norm",  # 忽略 attention 内部的 norm
            "re:.*k_norm",
        ]

        recipe = AWQModifier(
            ignore=ignore,
            mappings=mappings,
            duo_scaling=True,
            config_groups={
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": bits,
                        "type": "int",
                        "symmetric": True,
                        "group_size": group_size,
                        "strategy": "group",
                        "dynamic": False,
                        "actorder": None,
                        "observer": "mse",
                    },
                }
            },
        )
    else:
        # 纯文本模型使用简单配置
        recipe = AWQModifier(
            ignore=["lm_head"],
            scheme=f"W{bits}A16",
            targets=["Linear"],
            duo_scaling=True,
        )

    return recipe


def quantize_model(args):
    """执行 AWQ 量化"""
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print(f'使用 GPU: {args.gpu}')

    print('=' * 60)
    print('AWQ 量化脚本（llm-compressor）')
    print('参考: https://github.com/Junfeng-Pan/Qwen3-AWQ')
    print('=' * 60)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f'模型路径不存在: {model_path}')

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # 检测模型类型
    model_type = detect_model_type(model_path)
    print(f'检测到模型类型: {model_type}')
    print(f'输入模型: {model_path}')
    print(f'输出路径: {output_path}')
    print(f'量化位数: {args.bits}-bit')
    print(f'分组大小: {args.group_size}')
    print(f'最大序列长度: {args.max_seq_length}')

    # 步骤 1: 加载模型和处理器
    print('\n步骤 1: 加载模型...')

    # 尝试导入 swift（可选，用于注册自定义模型类型）
    try:
        import swift
        print('✓ 已加载 ms-swift')
    except (ImportError, Exception) as e:
        print(f'⚠ ms-swift 未安装或不兼容，跳过: {e}')
        print('  如果使用 Qwen3-VL 标准模型，不需要 ms-swift')

    if model_type in ('qwen3_vl', 'qwen2_vl', 'multimodal'):
        # 多模态模型使用 AutoModelForVision2Seq（需要 transformers >= 4.57.0）
        print(f'加载多模态模型...')
        try:
            from transformers import AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(
                str(model_path),
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
        except ImportError:
            # 回退：使用专用模型类
            print('  AutoModelForVision2Seq 不可用，尝试使用专用模型类...')
            if model_type == 'qwen3_vl':
                from transformers import Qwen3VLForConditionalGeneration
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    str(model_path),
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True,
                )
            elif model_type == 'qwen2_vl':
                from transformers import Qwen2VLForConditionalGeneration
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    str(model_path),
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    str(model_path),
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True,
                )

        # 加载处理器（回退到 AutoTokenizer）
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
            print('✓ 使用 AutoProcessor')
        except (ImportError, Exception):
            from transformers import AutoTokenizer
            processor = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            print('✓ AutoProcessor 不可用，使用 AutoTokenizer')

        print('✓ 多模态模型加载完成')
    else:
        # 纯文本模型
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f'使用 AutoModelForCausalLM 加载纯文本模型...')
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        print('✓ 纯文本模型加载完成')

    # 步骤 2: 配置 AWQ 量化配方
    print('\n步骤 2: 配置 AWQ 量化配方...')
    recipe = create_awq_recipe(model_type, bits=args.bits, group_size=args.group_size)
    print('✓ AWQ 配方配置完成')
    if model_type in ('qwen3_vl', 'qwen2_vl', 'multimodal'):
        print('  - 视觉编码器: 忽略（保持 FP16）')
        print('  - 语言模型: AWQ 4-bit 量化')
        print('  - q_norm/k_norm: 忽略')

    # 步骤 3: 加载校准数据集
    print('\n步骤 3: 加载校准数据集...')
    if args.dataset is not None:
        dataset_path = Path(args.dataset)
        if dataset_path.exists() and dataset_path.is_file():
            from datasets import load_dataset
            ds = load_dataset("json", data_files=str(dataset_path), split="train")
            ds = ds.shuffle(seed=42).select(range(min(args.num_calibration_samples, len(ds))))

            # 预处理函数
            def preprocess_function(example):
                if "messages" in example:
                    messages = example["messages"]
                elif "query" in example and "response" in example:
                    messages = [{"role": "user", "content": [{"type": "text", "text": example["query"]}]},
                                {"role": "assistant", "content": [{"type": "text", "text": example["response"]}]}]
                elif "text" in example:
                    messages = [{"role": "user", "content": [{"type": "text", "text": example["text"]}]}]
                else:
                    content = str(example)
                    messages = [{"role": "user", "content": [{"type": "text", "text": content}]}]

                # 确保 content 是列表格式
                for msg in messages:
                    if isinstance(msg["content"], str):
                        msg["content"] = [{"type": "text", "text": msg["content"]}]

                # 使用 processor 处理
                if hasattr(processor, 'apply_chat_template'):
                    text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    return processor(
                        text=[text],
                        padding=False,
                        max_length=args.max_seq_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                else:
                    # 纯文本 tokenizer
                    text = " ".join([msg["content"][0]["text"] if isinstance(msg["content"], list) else msg["content"] for msg in messages])
                    return processor(text, truncation=True, max_length=args.max_seq_length, return_tensors="pt")

            # 数据整理函数
            import torch
            def data_collator(batch):
                processed_batch = [preprocess_function(item) for item in batch]
                return {
                    "input_ids": torch.cat([x['input_ids'] for x in processed_batch], dim=0),
                    "attention_mask": torch.cat([x['attention_mask'] for x in processed_batch], dim=0),
                }

            dataset = ds
            print(f'✓ 加载本地数据集: {dataset_path} ({len(ds)} 条)')
        else:
            raise FileNotFoundError(f'数据集文件不存在: {dataset_path}')
    else:
        raise ValueError('必须指定校准数据集路径 --dataset')

    # 步骤 4: 执行量化
    print('\n步骤 4: 执行量化（这可能需要较长时间）...')

    # Monkey-patch: 绕过 torch.accelerator.get_memory_info 兼容性问题
    try:
        import compressed_tensors.offload.dispatch as dispatch_module
        def patched_get_device_memory():
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                return {
                    torch.device("cuda", i): torch.cuda.get_device_properties(i).total_memory
                    for i in range(device_count)
                }
            return {torch.device("cpu"): 0}
        dispatch_module.get_device_memory = patched_get_device_memory
    except Exception:
        pass

    from llmcompressor import oneshot

    oneshot(
        model=model,
        processor=processor,
        recipe=recipe,
        dataset=dataset,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
        data_collator=data_collator if args.dataset else None,
    )

    # 步骤 5: 保存量化模型
    print('\n步骤 5: 保存量化模型...')
    import torch
    model.to("cpu")
    torch.cuda.empty_cache()

    # 保存模型（使用压缩格式）
    model.save_pretrained(str(output_path), save_compressed=True, max_shard_size=args.max_shard_size)
    processor.save_pretrained(str(output_path))
    print('✓ 模型保存完成')

    # 显示保存的文件
    print('\n保存的文件:')
    for f in sorted(Path(output_path).iterdir()):
        if f.is_file():
            size = f.stat().st_size / (1024 * 1024)
            if size > 0.01:
                print(f'  {f.name}: {size:.2f} MB')

    # 复制配置文件
    if args.copy_config and args.official_model:
        print('\n步骤 6: 复制配置文件...')
        copy_config_files(args.official_model, output_path)

    # 保存量化信息
    quant_info = {
        'model_path': str(model_path),
        'output_path': str(output_path),
        'quant_method': 'awq',
        'quant_bits': args.bits,
        'group_size': args.group_size,
        'model_type': model_type,
        'dataset': args.dataset,
        'max_seq_length': args.max_seq_length,
    }
    info_path = output_path / 'quant_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(quant_info, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 60)
    print('量化完成！')
    print('=' * 60)
    print(f'量化模型保存到: {output_path}')

    total_size = 0
    for file in output_path.iterdir():
        if file.is_file():
            size = file.stat().st_size
            total_size += size

    print(f'总大小: {total_size / 1024 / 1024:.2f} MB')
    print(f'\n使用 vLLM 加载量化模型:')
    print(f'  vllm serve {output_path} --quantization awq --max-model-len 8192 --port 7890')

    return output_path


def main():
    args = parse_args()
    quantize_model(args)


if __name__ == '__main__':
    main()
