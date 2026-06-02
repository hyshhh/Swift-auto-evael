#!/bin/bash
# AWQ 量化启动脚本（支持 Qwen3-VL 多模态模型）

set -e

# ==================== 配置区 ====================
# 修改以下路径为你的实际路径

MERGED_MODEL="/media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-4B-Instruct"
OFFICIAL_MODEL="/media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-4B-Instruct"
AWQ_MODEL="/media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-4B-AWQ"

# 校准数据集
DATASET_PATH="/media/ddc/新加卷/hys/qmy/agent/data/sft_val.jsonl"

# 量化参数
BITS=4
GROUP_SIZE=128
GPU=0

# ==================== 环境检查 ====================
echo "=========================================="
echo "AWQ 量化 - Qwen3-VL 多模态模型"
echo "参考: https://docs.vllm.ai/en/latest/features/quantization/auto_awq.html"
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
echo "  1) AutoAWQ（推荐，vLLM 原生支持）"
echo "  2) llmcompressor"
read -p "请输入选择 [1/2]: " CHOICE

case $CHOICE in
    2)
        echo "使用 llmcompressor 量化..."
        CUDA_VISIBLE_DEVICES=$GPU python quantize_awq.py \
            --model $MERGED_MODEL \
            --output $AWQ_MODEL \
            --dataset $DATASET_PATH \
            --bits $BITS \
            --group_size $GROUP_SIZE \
            --copy_config \
            --official_model $OFFICIAL_MODEL
        ;;
    *)
        echo "使用 AutoAWQ 量化..."
        python quantize_autoawq.py \
            --model $MERGED_MODEL \
            --output $AWQ_MODEL \
            --dataset $DATASET_PATH \
            --bits $BITS \
            --group_size $GROUP_SIZE \
            --version GEMM \
            --copy_config \
            --official_model $OFFICIAL_MODEL \
            --gpu $GPU
        ;;
esac

echo ""
echo "=========================================="
echo "✓ 量化完成!"
echo "输出路径: $AWQ_MODEL"
echo ""
echo "使用 vLLM 加载量化模型:"
echo "  vllm serve $AWQ_MODEL \\"
echo "      --api-key abc123 \\"
echo "      --served-model-name Qwen/Qwen3-VL-4B-AWQ \\"
echo "      --max-model-len 8192 \\"
echo "      --port 7890 \\"
echo "      --quantization awq"
echo "=========================================="
