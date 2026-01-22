#!/bin/bash

ddpm_sampling=False

in_channels=1
batch_size=64
save_interval=5000

num_classes=2
image_size=128
threshold=0.1
version=dilated

DATA_ROOT="./data"
LOG_ROOT="./logs"

export OPENAI_LOGDIR="${LOG_ROOT}/logs_atlas_normal_99_11_128/logs_guided_${threshold}_${version}_t1"
mkdir -p "$OPENAI_LOGDIR"

data_dir="${DATA_ROOT}/ATLAS_2/preprocessed_data_t1_00_128"
image_dir="$OPENAI_LOGDIR/images"
mkdir -p "$image_dir"

GUI_FLAGS="--w 1 1.8 2 3 --threshold $threshold"

DATA_FLAGS="--image_size $image_size --num_classes $num_classes --class_cond True --ret_lab True --mixed True"

MODEL_FLAGS="--unet_ver $version --clf_free True\
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

TRAIN_FLAGS="--data_dir $data_dir --image_dir $image_dir --batch_size $batch_size --ddpm_sampling $ddpm_sampling --total_epochs 1000"

EVA_FLAGS="--save_interval $save_interval --sample_shape 12 $in_channels $image_size $image_size"

export MASTER_ADDR=localhost
export MASTER_PORT=12368

NUM_GPUS=1
torchrun --nproc-per-node $NUM_GPUS \
         --nnodes=1\
         --rdzv-backend=c10d\
         --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
        ./scripts/train_dilated.py --name atlas $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $GUI_FLAGS $EVA_FLAGS

