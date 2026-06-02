#!/bin/bash
# AWQ 量化启动脚本

set -e

# ==================== 配置区 ====================
# 修改以下路径为你的实际路径

MODEL_PATH="/media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b"
OUTPUT_PATH="/media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-awq"
DATASET_PATH="/media/ddc/新加卷/hys/qmy/agent/data/sft_val.jsonl"
OFFICIAL_MODEL="/media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B"

# 量化参数
BITS=4
GROUP_SIZE=128
GPU=0

# ==================== 环境检查 ====================
echo "=========================================="
echo "AWQ 量化 - Qwen3.5 多模态模型"
echo "=========================================="

# 检查 conda 环境
if [[ "$CONDA_DEFAULT_ENV" != "llmpress" ]]; then
    echo "⚠ 建议使用 llmpress 环境: conda activate llmpress"
fi

# 检查 GPU
echo "GPU 设备: $GPU"
nvidia-smi -i $GPU --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "⚠ 无法获取 GPU 信息"

# ==================== 量化方式选择 ====================
echo ""
echo "选择量化方式:"
echo "  1) llmcompressor (推荐)"
echo "  2) AutoAWQ"
read -p "请输入选择 [1/2]: " CHOICE

case $CHOICE in
    2)
        echo "使用 AutoAWQ 量化..."
        python quantize_autoawq.py \
            --model $MODEL_PATH \
            --output $OUTPUT_PATH \
            --dataset $DATASET_PATH \
            --bits $BITS \
            --group_size $GROUP_SIZE \
            --copy_config \
            --official_model $OFFICIAL_MODEL \
            --gpu $GPU
        ;;
    *)
        echo "使用 llmcompressor 量化..."
        CUDA_VISIBLE_DEVICES=$GPU python quantize_llmcompressor.py \
            --model $MODEL_PATH \
            --output $OUTPUT_PATH \
            --bits $BITS \
            --group_size $GROUP_SIZE \
            --dataset $DATASET_PATH \
            --copy_config \
            --official_model $OFFICIAL_MODEL
        ;;
esac

echo ""
echo "✓ 量化完成!"
echo "输出路径: $OUTPUT_PATH"
echo ""
echo "验证模型: python verify.py --model $OUTPUT_PATH"
