#!/bin/bash

# 此脚本已修改为单机单卡 Linux 环境运行，无 SLURM 依赖。
# 运行前请先激活 conda 环境，例如：`conda activate your_env_name`
# 或者取消下面注释以使用已有环境：
# source activate torch

# --- 路径配置 ---
DATA_ROOT="./data"   # 请修改为你的数据根目录
LOG_ROOT="./logs"    # 请修改为你的日志/权重保存目录

# --- 分布式参数（单机单卡） ---
export MASTER_ADDR=localhost
export MASTER_PORT=12355  # 使用不冲突的端口

# --- 模型与训练参数 ---
num_classes=2
image_size=128
version=v1
in_channels=4

# --- 日志与输出目录 ---
export OPENAI_LOGDIR="${LOG_ROOT}/clf"
mkdir -p "$OPENAI_LOGDIR"
echo "日志将保存在: $OPENAI_LOGDIR"

image_dir="$OPENAI_LOGDIR/images"
mkdir -p "$image_dir"

# --- 数据目录 ---
data_dir="${DATA_ROOT}/preprocessed_data"  # 请根据实际数据子路径调整

CLASSIFIER_FLAGS="--unet_ver $version --image_size $image_size --classifier_attention_resolutions 32,16,8 \
                --in_channels $in_channels --out_channels $num_classes \
                --classifier_depth 2 --classifier_width 128 \
                --classifier_pool attention \
                --classifier_resblock_updown True --classifier_use_scale_shift_norm True\
                --data_dir $data_dir --image_dir $image_dir --batch_size 64 --iterations 250000"

NUM_GPUS=1
torchrun --nproc-per-node $NUM_GPUS \
         --nnodes=1 \
         --rdzv-backend=c10d \
         --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
         ./scripts/train_classifier.py --name brats --save_interval 10000 \
         $CLASSIFIER_FLAGS

echo "脚本执行完成。"