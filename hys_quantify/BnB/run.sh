#!/bin/bash
# BnB NF4 量化启动脚本 - 适配 Qwen3-VL-4B
# 参考: https://docs.vllm.com.cn/en/latest/features/quantization/int4/

set -e

# ==================== 配置区 ====================
# 模型路径（Qwen3-VL-4B）
MERGED_MODEL="/media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-4B-Instruct"
OFFICIAL_MODEL="/media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-4B-Instruct"
BnB_MODEL="/media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-4B-BnB-NF4"

# 量化参数
BITS=4
DOUBLE_QUANT=true
COMPUTE_DTYPE="bfloat16"
MAX_SHARD_SIZE="2GB"

# GPU 配置
CUDA_VISIBLE_DEVICES=0

# ==================== 环境检查 ====================
echo "=========================================="
echo "BnB NF4 量化 - Qwen3-VL-4B 多模态模型"
echo "=========================================="
echo "原始模型: $MERGED_MODEL"
echo "输出路径: $BnB_MODEL"
echo "量化位数: ${BITS}-bit NF4"
echo "双重量化: $DOUBLE_QUANT"
echo "计算精度: $COMPUTE_DTYPE"
echo "=========================================="

# 检查模型是否存在
if [ ! -d "$MERGED_MODEL" ]; then
    echo "错误: 模型路径不存在: $MERGED_MODEL"
    exit 1
fi

# 检查 conda 环境
if [[ "$CONDA_DEFAULT_ENV" != "bnb" ]]; then
    echo "⚠ 建议使用 bnb 环境: conda activate bnb"
fi

# 检查 GPU
echo "GPU 设备:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "⚠ 无法获取 GPU 信息"

# ==================== 执行量化 ====================
cd "$(dirname "$0")"

if [[ "$DOUBLE_QUANT" == "true" ]]; then
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python quantize_bnb.py \
        --model $MERGED_MODEL \
        --output $BnB_MODEL \
        --bits $BITS \
        --double_quant \
        --compute_dtype $COMPUTE_DTYPE \
        --max_shard_size $MAX_SHARD_SIZE \
        --use_safetensors
else
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python quantize_bnb.py \
        --model $MERGED_MODEL \
        --output $BnB_MODEL \
        --bits $BITS \
        --compute_dtype $COMPUTE_DTYPE \
        --max_shard_size $MAX_SHARD_SIZE \
        --use_safetensors
fi

echo ""
echo "=========================================="
echo "量化完成！"
echo "=========================================="
echo "量化模型: $BnB_MODEL"
echo ""
echo "--- 使用方式 ---"
echo ""
echo "1. 验证量化模型:"
echo "   python verify_bnb.py --model $BnB_MODEL --bits 4"
echo ""
echo "2. 使用 vLLM 部署（推荐）:"
echo "   vllm serve $BnB_MODEL \\"
echo "       --quantization bitsandbytes \\"
echo "       --load-format bitsandbytes \\"
echo "       --dtype bfloat16 \\"
echo "       --max-model-len 8192 \\"
echo "       --port 7890 \\"
echo "       --api-key abc123 \\"
echo "       --served-model-name Qwen/Qwen3-VL-4B-BnB"
echo ""
echo "3. 使用 transformers 推理:"
echo "   python verify_bnb.py --model $BnB_MODEL --bits 4 --text '你好'"
echo ""
echo "4. QLoRA 微调:"
echo "   swift sft --model $BnB_MODEL --train_type lora ..."
echo "=========================================="
