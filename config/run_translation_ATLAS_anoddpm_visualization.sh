#!/bin/bash

DATA_ROOT="./data"
LOG_ROOT="./logs"

export MASTER_ADDR=localhost
export MASTER_PORT=12371

num_classes=0
seed=0
in_channels=1
num_channels=128
image_size=128
diffusion_steps=1000
model_num=156100
noise_type=simplex
use_ddpm=True

visualization_output_dir="./visualization_outputs"
num_samples_to_visualize=20

for round in 1
do
    for sample_steps in 200
    do
        export OPENAI_LOGDIR="${LOG_ROOT}/logs_atlas/translation_anoddpm_visualization_${noise_type}_${sample_steps}_${round}"
        mkdir -p "$OPENAI_LOGDIR"

        vis_output_dir="${visualization_output_dir}/atlas_anoddpm_vis_${noise_type}_${sample_steps}_${round}"
        mkdir -p "$vis_output_dir"

        data_dir="${DATA_ROOT}/ATLAS_2/preprocessed_data_t1_00_128"
        model_dir="${LOG_ROOT}/logs_atlas_normal_99_11_128/logs_anoddpm_${noise_type}"
        image_dir="$OPENAI_LOGDIR"

        if [ ! -d "$model_dir" ]; then
            echo "错误: 预训练模型目录不存在: $model_dir"
            exit 1
        fi

        MODEL_FLAGS="--unet_ver v1 --image_size $image_size --num_classes $num_classes \
                        --in_channels $in_channels --dropout 0 --attention_resolutions 32,16,8 \
                        --num_channels $num_channels --model_num $model_num --ema True \
                        --noise_type $noise_type"

        DATA_FLAGS="--batch_size 20 --num_batches 1 \
                    --batch_size_val 20 --num_batches_val 1 \
                    --modality 0 --seed $seed --use_weighted_sampler False"

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

        NUM_GPUS=1
        torchrun --nproc-per-node $NUM_GPUS \
                    --nnodes=1 \
                    --rdzv-backend=c10d \
                    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
                ./scripts/translation_HEALTHY_visualization.py --name atlas $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS $VISUALIZATION_FLAGS
    done
done

