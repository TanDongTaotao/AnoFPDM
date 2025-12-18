#!/bin/bash

DATA_ROOT="./data"
LOG_ROOT="./logs"

export MASTER_ADDR=localhost
export MASTER_PORT=12370

threshold=0.1
num_classes=2
seed=0
in_channels=1
num_channels=128
image_size=128
forward_steps=600
diffusion_steps=1000
model_num=290000
version=dilated

d_reverse=True

enable_dual_threshold=false
low_quant_offset=-0.15
high_quant_offset=0.15
entropy_weight=0.3
entropy_threshold=0.5

enable_snr_weighting=false
snr_smoothing=0.02
temporal_decay=0.95
min_weight=0.7
max_weight=1.2
consistency_weight=0.8
sensitivity_weight=0.2
aggregation_mode="robust_weighted"

visualization_output_dir="./visualization_outputs"
num_samples_to_visualize=5

w=30
for round in 1
do
    export OPENAI_LOGDIR="${LOG_ROOT}/logs_atlas_aba/translation_fpdm_dilated_visualization_${w}_${model_num}_${forward_steps}_${round}_x1"
    mkdir -p "$OPENAI_LOGDIR"

    vis_output_dir="${visualization_output_dir}/atlas_fpdm_dilated_vis_${w}_${model_num}_${forward_steps}_${round}"
    mkdir -p "$vis_output_dir"

    data_dir="${DATA_ROOT}/ATLAS_2/preprocessed_data_t1_00_128"
    model_dir="${LOG_ROOT}/logs_atlas_normal_99_11_128/logs_guided_${threshold}_${version}_t1"
    image_dir="$OPENAI_LOGDIR"

    if [ ! -d "$model_dir" ]; then
        echo "错误: 预训练模型目录不存在: $model_dir"
        exit 1
    fi

    MODEL_FLAGS="--image_size $image_size --num_classes $num_classes --in_channels $in_channels  \
                    --w $w --attention_resolutions 32,16,8 \
                    --num_channels $num_channels --model_num $model_num --ema True\
                    --forward_steps $forward_steps --d_reverse $d_reverse --unet_ver $version"

    DATA_FLAGS="--batch_size 10 --num_batches 1 \
                --batch_size_val 10 --num_batches_val 1\
                --modality 0 --use_weighted_sampler False --seed $seed"

    DIFFUSION_FLAGS="--null True \
                        --dynamic_clip False \
                        --diffusion_steps $diffusion_steps \
                        --noise_schedule linear \
                        --rescale_learned_sigmas False --rescale_timesteps False"

    DIR_FLAGS="--save_data False --data_dir $data_dir  --image_dir $image_dir --model_dir $model_dir"

    ABLATION_FLAGS="--last_only False --subset_interval -1 --t_e_ratio 1 --use_gradient_sam False --use_gradient_para_sam False"

    DUAL_THRESHOLD_FLAGS=""
    if [ "$enable_dual_threshold" = "true" ]; then
        DUAL_THRESHOLD_FLAGS="--enable_dual_threshold --low_quant_offset $low_quant_offset --high_quant_offset $high_quant_offset --entropy_weight $entropy_weight --entropy_threshold $entropy_threshold"
    fi

    SNR_WEIGHTING_FLAGS=""
    if [ "$enable_snr_weighting" = "true" ]; then
        SNR_WEIGHTING_FLAGS="--enable_snr_weighting --snr_smoothing $snr_smoothing --temporal_decay $temporal_decay --min_weight $min_weight --max_weight $max_weight --consistency_weight $consistency_weight --sensitivity_weight $sensitivity_weight --aggregation_mode $aggregation_mode"
    fi

    VISUALIZATION_FLAGS="--visualization_output_dir $vis_output_dir --num_samples_to_visualize $num_samples_to_visualize"

    NUM_GPUS=1
    torchrun --nproc-per-node $NUM_GPUS \
                --nnodes=1\
                --rdzv-backend=c10d\
                --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
            ./scripts/translation_FPDM_dilated_visualization.py --name atlas $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS $ABLATION_FLAGS $DUAL_THRESHOLD_FLAGS $SNR_WEIGHTING_FLAGS $VISUALIZATION_FLAGS
done

