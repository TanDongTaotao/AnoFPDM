#!/bin/bash

DATA_ROOT="./data"
LOG_ROOT="./logs"

export MASTER_ADDR=localhost
export MASTER_PORT=12371

threshold=0.1
image_size=128
num_classes=2
in_channels=1
num_channels=128
seed=0

diffusion_steps=1000
version=v2

visualization_output_dir="./visualization_outputs"
num_samples_to_visualize=5

for model_num in 290000
do
    for w in 1.3
    do
        for sample_steps in 300
        do
            export OPENAI_LOGDIR="${LOG_ROOT}/logs_atlas/translation_ddib_visualization_${w}_${sample_steps}_${model_num}"
            mkdir -p "$OPENAI_LOGDIR"

            vis_output_dir="${visualization_output_dir}/atlas_ddib_vis_${w}_${sample_steps}_${model_num}"
            mkdir -p "$vis_output_dir"

            data_dir="${DATA_ROOT}/ATLAS_2/preprocessed_data_t1_00_128"
            model_dir="${LOG_ROOT}/logs_atlas_normal_99_11_128/logs_guided_${threshold}_${version}_t1"
            image_dir="$OPENAI_LOGDIR"

            if [ ! -d "$model_dir" ]; then
                echo "错误: 预训练模型目录不存在: $model_dir"
                exit 1
            fi

            MODEL_FLAGS="--unet_ver $version --image_size $image_size --num_classes $num_classes \
                        --in_channels $in_channels --clf_free True \
                        --w $w --attention_resolutions 32,16,8 \
                        --num_channels $num_channels --model_num $model_num --ema True \
                        --learn_sigma False"

            DATA_FLAGS="--batch_size 10 --num_batches 1 \
                        --batch_size_val 10 --num_batches_val 1 \
                        --modality 0 --use_weighted_sampler False --seed $seed"

            DIFFUSION_FLAGS="--diffusion_steps $diffusion_steps \
                                --sample_steps $sample_steps \
                                --noise_schedule linear \
                                --rescale_learned_sigmas False \
                                --rescale_timesteps False \
                                --dynamic_clip False"

            DIR_FLAGS="--save_data False --data_dir $data_dir \
                        --image_dir $image_dir --model_dir $model_dir"

            VISUALIZATION_FLAGS="--visualization_output_dir $vis_output_dir --num_samples_to_visualize $num_samples_to_visualize"

            NUM_GPUS=1
            torchrun --nproc-per-node $NUM_GPUS \
                    --nnodes=1 \
                    --rdzv-backend=c10d \
                    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
                    ./scripts/translation_DDIB_visualization.py --name atlas $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS $VISUALIZATION_FLAGS
        done
    done
done

