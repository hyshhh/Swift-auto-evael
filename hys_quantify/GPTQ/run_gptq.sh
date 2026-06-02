#!/bin/bash
# GPTQ 量化运行脚本
# 用法: bash run_quantize_gptq.sh

# ==================== 依赖安装 ====================
# pip install gptqmodel optimum accelerate

# ==================== 配置区域 ====================

# 模型路径
MERGED_MODEL="/media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-4B-Instruct"
OFFICIAL_MODEL="/media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-4B-Instruct"
GPTQ_MODEL="/media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-4B-GPTQ"

# 量化参数
QUANT_BITS=4
GROUP_SIZE=128
DATASET="/media/ddc/新加卷/hys/qmy/agent/data/sft_val.jsonl"

# GPU 配置
CUDA_VISIBLE_DEVICES=0

# ==================== 执行区域 ====================

echo "=========================================="
echo "GPTQ 量化脚本（GPTQModel）"
echo "=========================================="
echo "合并模型: $MERGED_MODEL"
echo "官方模型: $OFFICIAL_MODEL"
echo "输出路径: $GPTQ_MODEL"
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

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python quantize_gptq.py \
    --model $MERGED_MODEL \
    --output $GPTQ_MODEL \
    --bits $QUANT_BITS \
    --group_size $GROUP_SIZE \
    --dataset $DATASET \
    --max_calib_samples 128 \
    --batch_size 1 \
    --copy_config \
    --official_model $OFFICIAL_MODEL

echo ""
echo "=========================================="
echo "量化完成！"
echo "=========================================="
echo "量化模型: $GPTQ_MODEL"
echo ""
echo "使用 vLLM 加载量化模型:"
echo "  CUDA_VISIBLE_DEVICES=1 vllm serve $GPTQ_MODEL \\"
echo "      --api-key abc123 \\"
echo "      --served-model-name Qwen/Qwen3-VL-4B \\"
echo "      --max-model-len 10240 \\"
echo "      --port 7890 \\"
echo "      --gpu-memory-utilization 0.3 \\"
echo "      --max-num-seqs 10 \\"
echo "      --enable-auto-tool-choice \\"
echo "      --tool-call-parser qwen3_xml \\"
echo "      --quantization gptq"
echo "=========================================="
