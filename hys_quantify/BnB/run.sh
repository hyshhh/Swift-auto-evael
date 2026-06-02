#!/bin/bash
# BnB NF4 量化启动脚本（支持流式保存）

set -e

# ==================== 配置区 ====================
# 修改以下路径为你的实际路径

MODEL_PATH="/media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b"
OUTPUT_PATH="/media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-bnb-nf4"

# 量化参数
BITS=4
DOUBLE_QUANT=true
COMPUTE_DTYPE="bfloat16"
MAX_SHARD_SIZE="1GB"  # 每个分片最大大小（2B 模型约 1-2GB）
SAVE_MERGED=false     # 是否保存合并模型（会变大）

# ==================== 环境检查 ====================
echo "=========================================="
echo "BnB NF4 量化 - Qwen3.5 多模态模型"
echo "=========================================="

# 检查 conda 环境
if [[ "$CONDA_DEFAULT_ENV" != "bnb" ]]; then
    echo "⚠ 建议使用 bnb 环境: conda activate bnb"
fi

# 检查 GPU
echo "GPU 设备:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "⚠ 无法获取 GPU 信息"

# ==================== 量化方式选择 ====================
echo ""
echo "选择量化类型:"
echo "  1) NF4 (推荐，压缩率 75%)"
echo "  2) INT8 (精度更高，压缩率 50%)"
read -p "请输入选择 [1/2]: " CHOICE

case $CHOICE in
    2)
        echo "使用 INT8 量化..."
        python quantize_bnb.py \
            --model $MODEL_PATH \
            --output $OUTPUT_PATH \
            --bits 8
        ;;
    *)
        echo "使用 NF4 量化..."
        if [[ "$DOUBLE_QUANT" == "true" ]]; then
            python quantize_bnb.py \
                --model $MODEL_PATH \
                --output $OUTPUT_PATH \
                --bits 4 \
                --double_quant \
                --compute_dtype $COMPUTE_DTYPE \
                --max_shard_size $MAX_SHARD_SIZE \
                --use_safetensors
        else
            python quantize_bnb.py \
                --model $MODEL_PATH \
                --output $OUTPUT_PATH \
                --bits 4 \
                --compute_dtype $COMPUTE_DTYPE \
                --max_shard_size $MAX_SHARD_SIZE \
                --use_safetensors
        fi
        ;;
esac

echo ""
echo "✓ 量化完成!"
echo "输出路径: $OUTPUT_PATH"
echo ""
echo "验证模型: python verify.py --model $OUTPUT_PATH --bits $BITS"
echo "多模态验证: python verify.py --model $OUTPUT_PATH --bits $BITS --image /path/to/image.jpg"
