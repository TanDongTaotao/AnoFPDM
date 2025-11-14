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
export MASTER_PORT=12358  # 使用不冲突的端口

# --- 训练参数 ---
in_channels=4
batch_size=32
save_interval=10000
num_classes=2  # healthy or unhealthy
image_size=128
version=v1     # UNet 版本

# --- 日志与输出目录 ---
export OPENAI_LOGDIR="${LOG_ROOT}/clf_guided"
mkdir -p "$OPENAI_LOGDIR"
echo "日志将保存在: $OPENAI_LOGDIR"

# 数据目录与图片输出目录
data_dir="${DATA_ROOT}/preprocessed_data"  # 请根据实际数据子路径调整
image_dir="$OPENAI_LOGDIR/images"
mkdir -p "$image_dir"

DATA_FLAGS="--image_size $image_size --num_classes $num_classes --class_cond True --ret_lab True --mixed True"

MODEL_FLAGS="--unet_ver $version \
             --in_channels $in_channels \
             --num_channels 128 \
             --attention_resolutions 32,16,8 \
             --learn_sigma True"

DIFFUSION_FLAGS="--clf_free False \
                 --diffusion_steps 1000 \
                 --noise_schedule linear \
                 --rescale_learned_sigmas False \
                 --rescale_timesteps False"

TRAIN_FLAGS="--data_dir $data_dir --image_dir $image_dir --batch_size $batch_size"

EVA_FLAGS="--save_interval $save_interval --sample_shape 12 $in_channels $image_size $image_size \
           --timestep_respacing ddim1000"

# 视觉检查相关（如不需要可置空）
GUI_FLAGS=""  # 这里保留占位，避免未定义变量报错

NUM_GPUS=1
torchrun --nproc-per-node $NUM_GPUS \
         --nnodes=1 \
         --rdzv-backend=c10d \
         --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
         ./scripts/train.py --name brats \
         $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $GUI_FLAGS $EVA_FLAGS

echo "脚本执行完成。"



