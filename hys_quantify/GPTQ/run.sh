#!/bin/bash
# GPTQ 量化启动脚本

set -e

# ==================== 配置区 ====================
# 修改以下路径为你的实际路径

MODEL_PATH="/media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b"
OUTPUT_PATH="/media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-gptq"
DATASET_PATH="/media/ddc/新加卷/hys/qmy/agent/data/sft_val.jsonl"
OFFICIAL_MODEL="/media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B"

# 量化参数
BITS=4
GROUP_SIZE=128
GPU=0

# ==================== 环境检查 ====================
echo "=========================================="
echo "GPTQ 量化 - Qwen3.5 多模态模型"
echo "=========================================="

# 检查 conda 环境
if [[ "$CONDA_DEFAULT_ENV" != "swifthys" ]]; then
    echo "⚠ 建议使用 swifthys 环境: conda activate swifthys"
fi

# 检查 GPU
echo "GPU 设备: $GPU"
nvidia-smi -i $GPU --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "⚠ 无法获取 GPU 信息"

# ==================== 量化方式选择 ====================
echo ""
echo "选择量化方式:"
echo "  1) run_gptq_swift.py (推荐，自动处理多模态配置)"
echo "  2) quantize_gptq.py (需要指定参数)"
read -p "请输入选择 [1/2]: " CHOICE

case $CHOICE in
    2)
        echo "使用 quantize_gptq.py 量化..."
        CUDA_VISIBLE_DEVICES=$GPU python quantize_gptq.py \
            --model $MODEL_PATH \
            --output $OUTPUT_PATH \
            --dataset $DATASET_PATH \
            --bits $BITS \
            --group_size $GROUP_SIZE \
            --copy_config \
            --official_model $OFFICIAL_MODEL
        ;;
    *)
        echo "使用 run_gptq_swift.py 量化..."
        python run_gptq_swift.py
        ;;
esac

echo ""
echo "✓ 量化完成!"
echo "输出路径: $OUTPUT_PATH"
echo ""
echo "验证模型: python verify.py --model $OUTPUT_PATH"
