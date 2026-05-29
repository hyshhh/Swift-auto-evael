#!/bin/bash
# AWQ 量化运行脚本
# 用法: bash run_quantize.sh

# ==================== 配置区域 ====================

# 模型路径
MERGED_MODEL="/media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b"
OFFICIAL_MODEL="/media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B"
AWQ_MODEL="/media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-awq"

# 量化参数
QUANT_BITS=4
GROUP_SIZE=128
DATASET="alpaca-en"  # 校准数据集，可改为你的数据集路径

# GPU 配置
CUDA_VISIBLE_DEVICES=0

# ==================== 执行区域 ====================

echo "=========================================="
echo "AWQ 量化脚本"
echo "=========================================="
echo "合并模型: $MERGED_MODEL"
echo "官方模型: $OFFICIAL_MODEL"
echo "输出路径: $AWQ_MODEL"
echo "量化位数: ${QUANT_BITS}-bit"
echo "分组大小: $GROUP_SIZE"
echo "校准数据: $DATASET"
echo "=========================================="

# 检查模型是否存在
if [ ! -d "$MERGED_MODEL" ]; then
    echo "错误: 合并模型路径不存在: $MERGED_MODEL"
    exit 1
fi

# 运行量化脚本
cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python quantize_awq.py \
    --model $MERGED_MODEL \
    --output $AWQ_MODEL \
    --bits $QUANT_BITS \
    --group_size $GROUP_SIZE \
    --dataset $DATASET \
    --copy_config \
    --official_model $OFFICIAL_MODEL

echo ""
echo "=========================================="
echo "量化完成！"
echo "=========================================="
echo "量化模型: $AWQ_MODEL"
echo ""
echo "使用 vLLM 加载量化模型:"
echo "  vllm serve $AWQ_MODEL \\"
echo "      --api-key abc123 \\"
echo "      --served-model-name Qwen/Qwen3-VL-4B-AWQ \\"
echo "      --max-model-len 8192 \\"
echo "      --port 7890 \\"
echo "      --quantization awq"
echo "=========================================="
