#!/bin/bash

# 此脚本为 anoddpm（不使用分类引导）的可视化推理脚本，已适配本地单机单卡环境。
# 运行前请确保已激活 conda 环境，例如：
#   conda activate your_env_name
# 或取消下面的注释：
# source activate torch

# --- [步骤 1] ---
# 设置数据和模型目录根路径（请按需修改）
DATA_ROOT="./brats21data"    # <-- [请修改] 指向您的数据集路径
LOG_ROOT="./logs"     # <-- [请修改] 指向您存放日志和模型的主目录

# --- [步骤 2] ---
# 分布式运行参数（单机单卡保持默认即可）
export MASTER_ADDR=localhost
export MASTER_PORT=12360  # 使用不同端口避免冲突

# --- [步骤 3] ---
# 模型与推理相关超参数
num_classes=0            # anoddpm 为无引导（不区分类），保持 0
seed=0                   # 数据加载与推理复现用随机种子
in_channels=4
num_channels=128
image_size=128
diffusion_steps=1000
model_num=100000         # 训练步数对应模型
noise_type=gaussian       # 可选：simplex 或 gaussian
use_ddpm=True            # 使用 DDPM 采样（True/False）

# 可视化参数
visualization_output_dir="./visualization_outputs"
num_samples_to_visualize=20

# --- [步骤 4] ---
# 循环运行（用于不同采样步数的消融对比）
for round in 1
do
    for sample_steps in 200
    do
        # 设置日志输出目录
        export OPENAI_LOGDIR="${LOG_ROOT}/logs_brats/translation_anoddpm_visualization_${noise_type}_${sample_steps}_${round}"
        mkdir -p "$OPENAI_LOGDIR"
        echo "日志将保存在: $OPENAI_LOGDIR"

        # 设置可视化输出目录
        vis_output_dir="${visualization_output_dir}/anoddpm_vis_${noise_type}_${sample_steps}_${round}"
        mkdir -p "$vis_output_dir"
        echo "可视化图像将保存在: $vis_output_dir"

        # 数据与预训练模型目录（请按需修改）
        data_dir="${DATA_ROOT}"
        model_dir="${LOG_ROOT}/logs_brats_normal_99_11_128/logs_anoddpm_simplex"
        image_dir="$OPENAI_LOGDIR"

        # 检查模型目录是否存在
        if [ ! -d "$model_dir" ]; then
            echo "错误: 预训练模型目录不存在: $model_dir"
            echo "请确保您已训练对应模型或调整 model_dir 指向正确路径。"
            exit 1
        fi

        # 组装传递给 python 的参数
        MODEL_FLAGS="--unet_ver v1 --image_size $image_size --num_classes $num_classes \
                        --in_channels $in_channels --dropout 0 --attention_resolutions 32,16,8 \
                        --num_channels $num_channels --model_num $model_num --ema True \
                        --noise_type $noise_type"

        DATA_FLAGS="--batch_size 20 --num_batches 1 \
                    --batch_size_val 20 --num_batches_val 1 \
                    --modality 0 3 --seed $seed --use_weighted_sampler False"

        DIFFUSION_FLAGS="--diffusion_steps $diffusion_steps \
                            --sample_steps $sample_steps \
                            --noise_schedule linear \
                            --rescale_learned_sigmas False \
                            --rescale_timesteps False \
                            --dynamic_clip False \
                            --use_ddpm $use_ddpm"

        DIR_FLAGS="--save_data False --data_dir $data_dir \
                    --image_dir $image_dir --model_dir $model_dir"

        VISUALIZATION_FLAGS="--visualization_output_dir $vis_output_dir --num_samples_to_visualize $num_samples_to_visualize"

        # --- [步骤 5] ---
        # 单机单卡运行可视化推理脚本
        NUM_GPUS=1
        torchrun --nproc-per-node $NUM_GPUS \
                    --nnodes=1 \
                    --rdzv-backend=c10d \
                    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
                ./scripts/translation_HEALTHY_visualization.py --name brats $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS $VISUALIZATION_FLAGS
    done
done

echo "anoddpm 可视化脚本执行完成。"