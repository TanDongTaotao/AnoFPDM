#!/bin/bash

# Windows compatible version for ATLAS domain adaptation UNet training
# Remove SLURM and module commands

# Activate conda environment
source activate torch

ddpm_sampling=False # ddim or ddpm

in_channels=1  # ATLAS uses single channel (T1 images)
batch_size=64
save_interval=5000

num_classes=2 
image_size=128
threshold=0.1
version=domain_adapt  # Use domain adaptation UNet

# Target domain adaptation parameters (Atlas only)
TARGET_NAME="atlas"
TARGET_DATA_DIR="F:/PycharmProjects/AnoFPDM/data/atlas"
PRETRAINED_MODEL_PATH="F:/PycharmProjects/AnoFPDM/results/brats/model_dilated_clf_free_guided/model_best.pt"
DOMAIN_ADAPT_WEIGHT=1.0
CONSISTENCY_WEIGHT=1.0
ENTROPY_WEIGHT=0.1
FEATURE_ALIGN_WEIGHT=0.5

enable_domain_adaptation=True
num_domains=2
domain_embedding_dim=64
target_domain=1  # 0 for source domain, 1 for target domain

# Pre-trained model path (BraTS trained weights)
pretrained_model_path="./logs/logs_brats_normal_99_11_128/logs_guided_${threshold}_dilated/model_best.pt"
freeze_backbone=True  # Freeze the main UNet, only train domain adaptation layers

# log dir
export OPENAI_LOGDIR="./logs/logs_atlas_normal_99_11_128/logs_guided_${threshold}_domain_adapt_t1"
# data directory
data_dir="./data/ATLAS/preprocessed_data_t1_00_128"
image_dir="$OPENAI_LOGDIR/images"

GUI_FLAGS="--w 1 1.8 2 3 --threshold $threshold" # select w for visual check only

DATA_FLAGS="--image_size $image_size --num_classes $num_classes --class_cond True --ret_lab True --mixed True"

MODEL_FLAGS="--unet_ver $version --clf_free True\
            --in_channels $in_channels \
             --num_channels 128 \
             --attention_resolutions 32,16,8 \
             --learn_sigma False\
             --dropout 0.1\
             --weight_decay 0"

# Domain adaptation flags
DOMAIN_ADAPT_FLAGS="--enable_domain_adaptation $enable_domain_adaptation \
                   --num_domains $num_domains \
                   --domain_embedding_dim $domain_embedding_dim \
                   --target_domain $target_domain \
                   --pretrained_model_path $pretrained_model_path \
                   --freeze_backbone $freeze_backbone"

DIFFUSION_FLAGS="--diffusion_steps 1000\
                    --noise_schedule linear \
                    --rescale_learned_sigmas False \
                    --rescale_timesteps False\
                    --noise_type gaussian"

TRAIN_FLAGS="--data_dir $data_dir --image_dir $image_dir --batch_size $batch_size --ddpm_sampling $ddpm_sampling --total_epochs 50"

EVA_FLAGS="--save_interval $save_interval --sample_shape 12 $in_channels $image_size $image_size"

# Single machine setup
export MASTER_ADDR=localhost
export MASTER_PORT=12361  # Use different port to avoid conflicts

NUM_GPUS=1
torchrun --nproc_per_node $NUM_GPUS \
        ./scripts/train_dilated_domain_adapt.py \
                            --name $TARGET_NAME \
                            --data_dir $TARGET_DATA_DIR \
                            --pretrained_model_path $PRETRAINED_MODEL_PATH \
                            --domain_adapt_weight $DOMAIN_ADAPT_WEIGHT \
                            --consistency_weight $CONSISTENCY_WEIGHT \
                            --entropy_weight $ENTROPY_WEIGHT \
                            --feature_align_weight $FEATURE_ALIGN_WEIGHT \
                            --enable_domain_adaptation $enable_domain_adaptation \
                            --num_domains $num_domains \
                            --domain_embedding_dim $domain_embedding_dim \
                            --freeze_main_network $freeze_backbone \
                            --ddpm_sampling $ddpm_sampling \
                            --in_channels $in_channels \
                            --batch_size $batch_size \
                            --save_interval $save_interval \
                            --num_classes $num_classes \
                            --image_size $image_size \
                            --threshold $threshold \
                            --version $version \
                            --log_dir $log_dir \
                            --image_dir $image_dir \
                            --gui $gui \
                            --data $data \
                            --model $model \
                            --diffusion $diffusion \
                            --train $train \
                            --eval $eval \
                            --total_epochs 50