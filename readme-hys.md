## 第一个命令：基础推理
    ## 使用 Transformers 引擎
    ## LoRA 权重动态叠加到基座模型
    ## 适合快速测试
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
    --adapters /media/ddc/新加卷/hys/qmy/ms-swift/output/v2-20260528-211924/checkpoint-31\
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048

## 第二个命令：vLLM 加速推理
    # 使用 vLLM 引擎（PagedAttention 技术）
    # --merge_lora true：LoRA 合并到基座模型，生成新权重
    # --vllm_max_model_len 8192：vLLM 支持的最大上下文长度
    # 适合部署和生产环境

CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
    --adapters /media/ddc/新加卷/hys/qmy/ms-swift/output/v2-20260528-211924/checkpoint-31\
    --stream true \
    --merge_lora true \
    --infer_backend vllm \
    --vllm_max_model_len 8192 \
    --temperature 0 \
    --max_new_tokens 2048

# 验证集--自定义QA
CUDA_VISIBLE_DEVICES=2 python eval_behavior.py \
        --model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
        --adapters /media/ddc/新加卷/hys/qmy/ms-swift/output/v2-20260528-211924/checkpoint-31 \
        --dataset /media/ddc/新加卷/hys/qmy/agent/data/sft_val.jsonl
        --batch_size 8





# 环境
conda activate swift

# 启动
CUDA_VISIBLE_DEVICES=1 \
swift sft \
    --model /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
    --tuner_type lora \
    --dataset /media/ddc/新加卷/hys/qmy/agent/data/sft_train.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 2 \
    --lora_alpha 8 \
    --target_modules all-linear \
    --gradient_accumulation_steps 32 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 6 \
    --model_author swift \
    --model_name swift-robot

# 启动2
# 配置参数
    --dataloader_num_workers 4  使用 4 个子进程并行加载数据
    
    --num_train_epochs 1        表示整个数据集遍历训练 1 轮。

    --per_device_train_batch_size 1 表示每张 GPU 一次处理 1 条数据

    --lora_rank 8：LoRA 的秩（矩阵的维度）  越大：拟合能力越强，但显存占用越多，越小：显存省，但可能学不到复杂模式

    --lora_alpha 32：缩放系数控制 LoRA 权重对原始模型的影响程度

    --target_modules all-linear 表示对所有线性层应用 LoRA。减少到 q_proj,v_proj 或 q_proj,k_proj,v_proj,o_proj
        --freeze_llm true           # 多模态模型：冻结 LLM 部分
        --freeze_vit true           # 多模态模型：冻结 ViT 部分
        --freeze_aligner true       # 多模态模型：冻结对齐器
        --modules_to_save embed_tokens lm_head #表示即使使用 LoRA 训练，也对这两个层进行全参数训练并保存权重。
    
    --gradient_accumulation_steps 16 表示每 16 步才更新一次模型权重。
    总 step 数 = 数据量 ÷ (per_device_train_batch_size × gradient_accumulation_steps × GPU数) × epochs

    --torch_dtype bfloat16 \加载模型时，把权重从原始精度（如 float32）转换为 bfloat16

# 训练时的参数
loss: 0.3411	损失值，越低越好，0.34 说明模型在快速学习
grad_norm: 7.841	梯度范数，用于监控梯度爆炸/消失，正常范围
learning_rate: 5e-05	当前学习率，因为 warmup 还没结束，从 0 逐渐增加到 1e-4
token_acc: 0.915	token 准确率 91.5%，模型预测正确的 token 比例
epoch: 0.03252	当前进度 3.25%，数据集还没遍历完一遍
global_step/max_steps: 1/31	第 1 步 / 共 31 步
elapsed_time: 41s	已训练 41 秒
remaining_time: 20m 35s	预计剩余 20 分 35 秒
memory(GiB): 10.7	显存占用 10.7 GiB
train_speed(s/it): 41.16	每步耗时 41.16 秒