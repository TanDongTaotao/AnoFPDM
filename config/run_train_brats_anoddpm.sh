#!/bin/bash

# Windows compatible version
# Remove SLURM and module commands

# Activate conda environment
source activate torch


noise_type=simplex
# noise_type=gaussian

in_channels=4
batch_size=4
save_interval=5000
num_classes=0 # unguided
image_size=128

# log directory
export OPENAI_LOGDIR="./logs/logs_brats_normal_99_11_128/logs_anoddpm_${noise_type}"
# data directory
data_dir="./data"
image_dir="$OPENAI_LOGDIR/images"

DATA_FLAGS="--image_size $image_size --num_classes $num_classes \
                --class_cond False --ret_lab False --mixed False
                --n_unhealthy_patients 0" # only train on non-tumour slices

MODEL_FLAGS="--unet_ver v1\
            --in_channels $in_channels \
             --num_channels 128 \
             --attention_resolutions 32,16,8 \
             --learn_sigma False\
             --dropout 0"

DIFFUSION_FLAGS="--diffusion_steps 1000\
                --noise_type $noise_type \
                    --noise_schedule linear \
                    --rescale_learned_sigmas False \
                    --rescale_timesteps False"

TRAIN_FLAGS="--data_dir $data_dir --image_dir $image_dir --batch_size $batch_size --total_epochs 1000"


EVA_FLAGS="--save_interval $save_interval --sample_shape 12 $in_channels $image_size $image_size --ddpm_sampling True"
            


# Single machine setup
export MASTER_ADDR=localhost
export MASTER_PORT=12345

NUM_GPUS=1
torchrun --nproc-per-node $NUM_GPUS \
        ./scripts/train.py --name brats \
                            $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $GUI_FLAGS $EVA_FLAGS



