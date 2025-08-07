#!/bin/bash

# Windows compatible version for HAE UNet Lite training
# Remove SLURM and module commands

# Activate conda environment
source activate torch

ddpm_sampling=False # ddim or ddpm

in_channels=4
batch_size=14
save_interval=5000

num_classes=2 
image_size=128
threshold=0.1
version=hae_lite  # Use HAE UNet Lite version

# log dir
export OPENAI_LOGDIR="./logs/clf_free_guided_hae_lite"
# data directory
data_dir="./data"
image_dir="$OPENAI_LOGDIR/images"

GUI_FLAGS="--w 1 1.8 2 3 --threshold $threshold" # select w for visual check only

DATA_FLAGS="--image_size $image_size --num_classes $num_classes --class_cond True --ret_lab True --mixed True"

MODEL_FLAGS="--unet_ver $version --clf_free True --use_hae True\
            --in_channels $in_channels \
             --num_channels 128 \
             --attention_resolutions 32,16,8 \
             --learn_sigma False\
             --dropout 0.1\
             --weight_decay 0"

DIFFUSION_FLAGS="--diffusion_steps 1000\
                    --noise_schedule linear \
                    --rescale_learned_sigmas False \
                    --rescale_timesteps False\
                    --noise_type gaussian"

TRAIN_FLAGS="--data_dir $data_dir --image_dir $image_dir --batch_size $batch_size --ddpm_sampling $ddpm_sampling --total_epochs 200"

EVA_FLAGS="--save_interval $save_interval --sample_shape 12 $in_channels $image_size $image_size"

# Single machine setup
export MASTER_ADDR=localhost
export MASTER_PORT=12360  # Different port to avoid conflicts

NUM_GPUS=1
torchrun --nproc-per-node $NUM_GPUS \
        ./scripts/train_hae_lite.py --name brats \
                            $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $GUI_FLAGS $EVA_FLAGS