#!/bin/bash

# 此脚本已修改为可在本地单机单卡 Linux 环境运行，无 SLURM 依赖。
# 运行前请先激活 conda 环境，例如：`conda activate your_env_name`
# 或取消下面注释以使用已有环境：
# source activate torch

# --- 路径配置 ---
DATA_ROOT="./data"   # 请修改为你的数据根目录
LOG_ROOT="./logs_brats"  # 请修改为你的日志/权重保存目录

# --- 分布式参数（单机单卡） ---
export MASTER_ADDR=localhost
export MASTER_PORT=12359  # 使用不冲突的端口

# --- 通用参数 ---
modality=all
suffix=00
image_size=128
version=v1
num_classes=2
in_channels=4
seed=0  # 仅用于数据加载

diffusion_steps=1000
model_num=500000  # 预训练 UNet 步数
clf_num=149999    # 预训练分类器步数

# 轮次与采样设置（可按需调整）
for round in 1  # 多次运行用于估计误差条
do
    for sample_steps in 200  # 可在验证集上网格搜索
    do  
        for classifier_scale in 1000  # 可在验证集上网格搜索
        do
            export OPENAI_LOGDIR="${LOG_ROOT}/translation_clf_guided_${round}_${sample_steps}_${classifier_scale}_plot"
            mkdir -p "$OPENAI_LOGDIR"
            echo "日志将保存在: $OPENAI_LOGDIR"

            data_dir="${DATA_ROOT}/BraTS21_training/preprocessed_data_all_00_128"
            model_dir="./trained_weights/clf-guided"
            classifier_dir="./trained_weights/clf"

            image_dir="$OPENAI_LOGDIR"

            # 目录存在性检查（可选）
            if [ ! -d "$model_dir" ]; then
                echo "错误: 预训练 UNet 目录不存在: $model_dir"
                echo "请确保已训练并保存了引导模型，或修改 model_dir 变量指向正确路径。"
                exit 1
            fi
            if [ ! -d "$classifier_dir" ]; then
                echo "错误: 预训练分类器目录不存在: $classifier_dir"
                echo "请确保已训练并保存了分类器，或修改 classifier_dir 变量指向正确路径。"
                exit 1
            fi

            MODEL_FLAGS="--unet_ver $version --image_size $image_size --clf_free False \
                        --num_classes $num_classes --in_channels $in_channels \
                        --classifier_scale $classifier_scale --dropout 0 \
                        --attention_resolutions 32,16,8 --learn_sigma True \
                        --num_channels 128 --model_num $model_num --ema True"

            CLASSIFIER_FLAGS="--clf_dir $classifier_dir --clf_num $clf_num \
                                --classifier_attention_resolutions 32,16,8 \
                                --out_channels $num_classes \
                                --classifier_depth 2 --classifier_width 128 \
                                --classifier_pool attention \
                                --classifier_resblock_updown True \
                                --classifier_use_scale_shift_norm True"

            DATA_FLAGS="--batch_size 100 --num_batches 2 \
                        --batch_size_val 100 --num_batches_val 10 \
                        --modality 0 3 --seed $seed --use_weighted_sampler False"

            DIFFUSION_FLAGS="--diffusion_steps $diffusion_steps \
                                --sample_steps $sample_steps \
                                --noise_schedule linear \
                                --rescale_learned_sigmas False \
                                --rescale_timesteps False"

            DIR_FLAGS="--save_data True --data_dir $data_dir \
                        --image_dir $image_dir --model_dir $model_dir"

            NUM_GPUS=1
            torchrun --nproc-per-node $NUM_GPUS \
                     --nnodes=1 \
                     --rdzv-backend=c10d \
                     --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
                     ./scripts/translation_CLF.py $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS $CLASSIFIER_FLAGS
        done
    done
done

echo "脚本执行完成。"