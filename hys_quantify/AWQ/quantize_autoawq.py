"""
AWQ 量化脚本 - 使用 AutoAWQ 进行 AWQ 量化（标准 AWQ 格式，vLLM 兼容）

用法：
    python quantize_autoawq.py \
        --model /path/to/merged_model \
        --output /path/to/awq_model \
        --dataset /path/to/calibration.jsonl \
        --bits 4 \
        --gpu 0,1

说明：
    支持 Qwen3-VL 多模态模型和纯文本模型。
    量化后模型可直接用于 vLLM 推理（--quantization awq）。
    参考：https://docs.vllm.ai/en/latest/features/quantization/auto_awq.html
"""
import os
import json
import shutil
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='AWQ 量化脚本（使用 AutoAWQ，vLLM 兼容）')
    parser.add_argument('--model', type=str, required=True, help='合并后的模型路径')
    parser.add_argument('--output', type=str, required=True, help='量化后模型保存路径')
    parser.add_argument('--bits', type=int, default=4, choices=[3, 4], help='量化位数（默认 4-bit）')
    parser.add_argument('--group_size', type=int, default=128, help='量化分组大小')
    parser.add_argument('--dataset', type=str, default=None, help='校准数据集 JSONL 路径（默认使用 pileval）')
    parser.add_argument('--copy_config', action='store_true', help='复制官方模型配置文件')
    parser.add_argument('--official_model', type=str, default=None, help='官方模型路径（用于复制配置）')
    parser.add_argument('--max_calib_samples', type=int, default=128, help='校准样本数量')
    parser.add_argument('--max_calib_seq_len', type=int, default=512, help='校准最大序列长度')
    parser.add_argument('--n_parallel_calib_samples', type=int, default=None, help='并行校准样本数（降低显存占用）')
    parser.add_argument('--gpu', type=str, default=None, help='指定使用的 GPU，如 "0" 或 "0,1" 或 "1,2,3"')
    parser.add_argument('--zero_point', action='store_true', default=True, help='使用零点量化（默认开启）')
    parser.add_argument('--version', type=str, default='GEMM', choices=['GEMM', 'Marlin'], help='量化内核版本')
    return parser.parse_args()


def detect_model_type(model_path):
    """检测模型类型，返回模型类型标识"""
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        model_type = config.get('model_type', '')

        # 检查是否为 Qwen3-VL 多模态模型
        if model_type == 'qwen3_vl':
            return 'qwen3_vl'
        elif model_type == 'qwen2_vl':
            return 'qwen2_vl'
        elif 'visual_config' in config or 'vision_config' in config:
            return 'multimodal'
        elif model_type == 'qwen3_5':
            return 'qwen3_5'
        elif 'qwen' in model_type:
            return 'qwen'

    return 'text_only'


def copy_config_files(official_model, output_path):
    """从官方模型复制配置文件"""
    official_model = Path(official_model)
    output_path = Path(output_path)

    # 多模态模型需要的配置文件（比纯文本模型更多）
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


def patch_qwen35_init_use_cache():
    """AutoAWQ may pass use_cache into the model constructor; Qwen3.5 does not accept it."""
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    except Exception as e:
        print(f'跳过 Qwen3.5 use_cache 兼容补丁: {e}')
        return

    if getattr(Qwen3_5ForCausalLM.__init__, '_hys_use_cache_patched', False):
        return

    old_init = Qwen3_5ForCausalLM.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.pop('use_cache', None)
        return old_init(self, *args, **kwargs)

    patched_init._hys_use_cache_patched = True
    Qwen3_5ForCausalLM.__init__ = patched_init
    print('已应用 Qwen3.5 use_cache 构造参数兼容补丁')



def patch_qwen35_awq_get_layers_for_scaling(awq_model):
    import torch
    from awq.models.qwen3 import Qwen3AWQForCausalLM
    hf_model = awq_model.model
    if not hasattr(hf_model, "model") or not hasattr(hf_model.model, "layers"):
        print("skip: no layers")
        return
    decoder_layers = hf_model.model.layers
    if len(decoder_layers) == 0:
        print("skip: empty layers")
        return
    first_layer = decoder_layers[0]
    layer_type = type(first_layer).__name__
    print(f"DecoderLayer type: {layer_type}")
    if hasattr(first_layer, "self_attn"):
        print("self_attn exists, no patch needed")
        return
    candidate_attrs = []
    for attr_name in dir(first_layer):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(first_layer, attr_name)
        except Exception:
            continue
        if not isinstance(attr, torch.nn.Module):
            continue
        for sub_name, sub_module in attr.named_modules():
            if "q_proj" in sub_name and isinstance(sub_module, torch.nn.Linear):
                candidate_attrs.append(attr_name)
                break
    if not candidate_attrs:
        print("warning: no q_proj found")
        return
    attn_attr = candidate_attrs[0]
    print(f"found attn attr: {attn_attr}")
    def patched_get_layers_for_scaling(self):
        layers = self.model.model.layers
        modules = []
        for layer in layers:
            attn = getattr(layer, attn_attr)
            modules.extend([attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj])
        return modules
    Qwen3AWQForCausalLM.get_layers_for_scaling = patched_get_layers_for_scaling
    print(f"patched get_layers_for_scaling with attr: {attn_attr}")
def patch_qwen3vl_awq_support():
    """
    为 AutoAWQ 注册 Qwen3-VL 多模态模型支持（实验性）

    注意：
    - AutoAWQ 官方不支持 Qwen3-VL，此为实验性补丁
    - AutoAWQ 已被官方弃用，推荐使用 llmcompressor 量化 Qwen3-VL
    - 参考：https://github.com/casper-hansen/AutoAWQ
    """
    try:
        from awq.models.auto import AWQ_CAUSAL_LM_MODEL_MAP
        from awq.models.base import TRANSFORMERS_AUTO_MAPPING_DICT
        from awq.models.qwen2_vl import Qwen2VLAWQForCausalLM

        # Qwen3-VL 尝试复用 Qwen2-VL 处理器（实验性，可能不兼容）
        if "qwen3_vl" not in AWQ_CAUSAL_LM_MODEL_MAP:
            AWQ_CAUSAL_LM_MODEL_MAP["qwen3_vl"] = Qwen2VLAWQForCausalLM
            TRANSFORMERS_AUTO_MAPPING_DICT["qwen3_vl"] = "AutoModelForVision2Seq"
            print('⚠ 已注册 qwen3_vl 到 AutoAWQ（实验性，复用 Qwen2-VL 处理器）')
            print('  如果量化失败，请使用 llmcompressor 方式: python quantize_awq.py')

        # Qwen2-VL 已有官方支持，确保注册
        if "qwen2_vl" not in AWQ_CAUSAL_LM_MODEL_MAP:
            AWQ_CAUSAL_LM_MODEL_MAP["qwen2_vl"] = Qwen2VLAWQForCausalLM
            TRANSFORMERS_AUTO_MAPPING_DICT["qwen2_vl"] = "AutoModelForVision2Seq"
            print('已注册 qwen2_vl 到 AutoAWQ')
    except Exception as e:
        print(f'注册 Qwen3-VL AWQ 支持时出错: {e}')
        print('建议使用 llmcompressor 方式量化 Qwen3-VL')


def quantize_model(args):
    """执行 AWQ 量化"""
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print(f'使用 GPU: {args.gpu}')

    import swift  # 注册 qwen3_5 等自定义模型类型到 transformers

    # 检测模型类型
    model_type = detect_model_type(args.model)
    print(f'检测到模型类型: {model_type}')

    # 根据模型类型进行补丁和注册
    if model_type in ('qwen3_vl', 'qwen2_vl', 'multimodal'):
        patch_qwen3vl_awq_support()
        if model_type == 'qwen3_vl':
            print('')
            print('⚠ 警告: Qwen3-VL 使用 AutoAWQ 量化为实验性功能')
            print('  AutoAWQ 官方不支持 Qwen3-VL，推荐使用 llmcompressor:')
            print('  python quantize_awq.py --model ... --output ...')
            print('')
    elif model_type == 'qwen3_5':
        patch_qwen35_init_use_cache()
        # AutoAWQ 不认识 qwen3_5，monkey-patch 注册复用 Qwen3 处理器
        from awq.models.auto import AWQ_CAUSAL_LM_MODEL_MAP
        from awq.models.base import TRANSFORMERS_AUTO_MAPPING_DICT
        if "qwen3_5" not in AWQ_CAUSAL_LM_MODEL_MAP:
            from awq.models.qwen3 import Qwen3AWQForCausalLM
            AWQ_CAUSAL_LM_MODEL_MAP["qwen3_5"] = Qwen3AWQForCausalLM
            TRANSFORMERS_AUTO_MAPPING_DICT["qwen3_5"] = "AutoModelForCausalLM"
            print('已注册 qwen3_5 到 AutoAWQ（复用 Qwen3 处理器）')

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print('=' * 60)
    print('AWQ 量化脚本（AutoAWQ，vLLM 兼容格式）')
    print('=' * 60)
    print(f'参考: https://docs.vllm.ai/en/latest/features/quantization/auto_awq.html')

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f'模型路径不存在: {model_path}')

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f'输入模型: {model_path}')
    print(f'输出路径: {output_path}')
    print(f'量化位数: {args.bits}-bit')
    print(f'分组大小: {args.group_size}')
    print(f'量化内核: {args.version}')

    # 量化配置
    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.group_size,
        "w_bit": args.bits,
        "version": args.version,
    }

    # 加载模型和分词器
    print('\n步骤 1: 加载模型...')
    model = AutoAWQForCausalLM.from_pretrained(
        str(model_path),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # 禁用缓存以节省内存
    if hasattr(model, 'config'):
        model.config.use_cache = False

    # 仅对 Qwen3.5 文本模型应用补丁
    if model_type == 'qwen3_5':
        patch_qwen35_awq_get_layers_for_scaling(model)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    # 加载校准数据集
    print('步骤 2: 加载校准数据集...')
    calib_data = None
    if args.dataset is not None:
        dataset_path = Path(args.dataset)
        if dataset_path.exists() and dataset_path.is_file():
            from datasets import load_dataset
            ds = load_dataset("json", data_files=str(dataset_path), split="train")
            ds = ds.shuffle(seed=42).select(range(min(args.max_calib_samples, len(ds))))

            def preprocess(example):
                return example.get("query", "") + " " + example.get("response", "")

            calib_data = [preprocess(example) for example in ds]
            print(f'加载本地数据集: {dataset_path} ({len(calib_data)} 条)')
        elif dataset_path.exists() and dataset_path.is_dir():
            raise ValueError(f'数据集路径是一个目录，需要指定 JSONL 文件: {dataset_path}')
        else:
            raise FileNotFoundError(f'数据集文件不存在: {dataset_path}')
    else:
        print('使用默认校准数据集: pileval')

    # 执行量化
    print('步骤 3: 执行量化...')
    quantize_kwargs = {
        "quant_config": quant_config,
        "max_calib_samples": args.max_calib_samples,
        "max_calib_seq_len": args.max_calib_seq_len,
    }
    if calib_data is not None:
        quantize_kwargs["calib_data"] = calib_data
    if args.n_parallel_calib_samples is not None:
        quantize_kwargs["n_parallel_calib_samples"] = args.n_parallel_calib_samples

    model.quantize(tokenizer, **quantize_kwargs)

    # 保存模型
    print('步骤 4: 保存量化模型...')
    model.save_quantized(str(output_path), safetensors=True)
    tokenizer.save_pretrained(str(output_path))

    # 保存 processor（多模态模型）
    if model_type in ('qwen3_vl', 'qwen2_vl', 'multimodal'):
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
            processor.save_pretrained(str(output_path))
            print('✓ Processor 保存完成')
        except Exception as e:
            print(f'⚠ Processor 保存失败: {e}')

    # 复制配置文件
    if args.copy_config and args.official_model:
        print('\n步骤 5: 复制配置文件...')
        copy_config_files(args.official_model, output_path)

    # 保存量化信息
    quant_info = {
        'model_path': str(model_path),
        'output_path': str(output_path),
        'quant_method': 'awq',
        'quant_bits': args.bits,
        'group_size': args.group_size,
        'version': args.version,
        'zero_point': args.zero_point,
        'model_type': model_type,
        'dataset': args.dataset or 'pileval',
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
            if size > 1024 * 1024:
                print(f'  {file.name}: {size / 1024 / 1024:.2f} MB')

    print(f'\n总大小: {total_size / 1024 / 1024:.2f} MB')

    return output_path


def main():
    args = parse_args()
    quantize_model(args)


if __name__ == '__main__':
    main()
