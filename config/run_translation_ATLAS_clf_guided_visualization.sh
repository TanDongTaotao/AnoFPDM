#!/bin/bash

DATA_ROOT="./data"
LOG_ROOT="./logs"

export MASTER_ADDR=localhost
export MASTER_PORT=12371

image_size=128
version=v1
num_classes=2
in_channels=1
seed=0

diffusion_steps=1000
model_num=145000
clf_num=040000

visualization_output_dir="./visualization_outputs"
num_samples_to_visualize=5

for round in 1
do
    for sample_steps in 200
    do
        for classifier_scale in 500
        do
            export OPENAI_LOGDIR="${LOG_ROOT}/logs_atlas/translation_clf_guided_visualization_${round}_${sample_steps}_${classifier_scale}"
            mkdir -p "$OPENAI_LOGDIR"

            vis_output_dir="${visualization_output_dir}/atlas_clf_guided_vis_${round}_${sample_steps}_${classifier_scale}"
            mkdir -p "$vis_output_dir"

            data_dir="${DATA_ROOT}/ATLAS_2/preprocessed_data_t1_00_128"
            model_dir="${LOG_ROOT}/logs_atlas_normal_99_11_128/logs_clf_guided_v1"
            classifier_dir="${LOG_ROOT}/logs_atlas_clf"
            image_dir="$OPENAI_LOGDIR"

            if [ ! -d "$model_dir" ]; then
                echo "错误: 预训练模型目录不存在: $model_dir"
                exit 1
            fi
            if [ ! -d "$classifier_dir" ]; then
                echo "错误: 分类器目录不存在: $classifier_dir"
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

            DATA_FLAGS="--batch_size 100 --num_batches 5 \
                        --batch_size_val 100 --num_batches_val 10 \
                        --modality 0 --use_weighted_sampler False --seed $seed"

            DIFFUSION_FLAGS="--diffusion_steps $diffusion_steps \
                                --sample_steps $sample_steps \
                                --noise_schedule linear \
                                --rescale_learned_sigmas False \
                                --rescale_timesteps False"

            DIR_FLAGS="--save_data True --data_dir $data_dir \
                        --image_dir $image_dir --model_dir $model_dir"

            VISUALIZATION_FLAGS="--visualization_output_dir $vis_output_dir --num_samples_to_visualize $num_samples_to_visualize"

            NUM_GPUS=1
            torchrun --nproc-per-node $NUM_GPUS \
                        --nnodes=1 \
                        --rdzv-backend=c10d \
                        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
                    ./scripts/translation_CLF_visualization.py --name atlas $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS $CLASSIFIER_FLAGS $VISUALIZATION_FLAGS
        done
    done
done

