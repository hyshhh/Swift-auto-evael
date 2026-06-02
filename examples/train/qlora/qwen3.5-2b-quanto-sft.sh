#!/bin/bash
# Qwen3.5-2B Quanto 4-bit QLoRA SFT 训练脚本
# Quanto 支持多种位宽: 2,4,8,float8

CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
MS_SWIFT_LOG_LEVEL='INFO' \
swift sft \
    --model Qwen/Qwen3.5-2B \
    --dataset AI-ModelScope/alpaca-gpt4-data-zh#500 \
              AI-ModelScope/alpaca-gpt4-data-en#500 \
              AI-ModelScope/LMSYS-Chat-1M-Alpaca#500 \
              swift/test_finetune_vl_dataset.jsonl \
    --train_type lora \
    --quant_method quanto \
    --quant_bits 4 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --group_by_length true \
    --output_dir output/Qwen3.5-2B-quanto-qlora-sft \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
