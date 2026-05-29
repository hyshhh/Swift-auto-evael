## 环境准备
    conda create -n llmpress python=3.10 -y
    conda activate llmpress
## 运行
    CUDA_VISIBLE_DEVICES=0 python quantize_awq.py \
        --model /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b \
        --output /media/ddc/新加卷/hys/hysnew3/model/wt-Qwen2b-awq \
        --bits 4 \
        --group_size 128 \
        --dataset alpaca \
        --copy_config \
        --official_model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B
