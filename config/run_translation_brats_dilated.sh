#!/bin/bash

# This script is modified to run on a local machine instead of a SLURM cluster.
# Before running, make sure your conda environment is activated.
# Example: `conda activate your_env_name`
# Or uncomment the line below:
# source activate torch_base

# --- [Step 1] ---
# Set root paths for data and model directories.
# Modify these paths to match your environment.
DATA_ROOT="./data" # <-- [Edit] Path to your dataset root
LOG_ROOT="./logs"  # <-- [Edit] Root directory for logs and checkpoints

# --- [Step 2] ---
# Configure networking parameters for distributed runs.
# For single-machine runs, the defaults are usually fine.
export MASTER_ADDR=localhost
export MASTER_PORT=12359 # Use a different port to avoid conflicts

# --- [Step 3] ---
# Set model and inference hyperparameters
threshold=0.1
num_classes=2
seed=0 # Only used for the data loader
in_channels=4
num_channels=128
image_size=128
forward_steps=600
diffusion_steps=1000
model_num=210000
version=dilated  # Use dilated UNet

d_reverse=True # Set True to use DDIM reverse (deterministic encoding)
               # Otherwise DDPM reverse will be used (stochastic encoding)

# --- [Dual-threshold strategy parameters] ---
# Set true to enable dual-threshold; false uses the original single-threshold method.
enable_dual_threshold=false
# Low-threshold offset (negative means a lower threshold, increasing recall)
low_quant_offset=-0.15
# High-threshold offset (positive means a higher threshold, increasing precision)
high_quant_offset=0.15
# Weight of local entropy in final mask fusion
entropy_weight=0.3
# Threshold for binarizing local entropy
entropy_threshold=0.5

# --- [SNR weighting strategy parameters] ---
# Set true to enable SNR-weighted aggregation; false uses the traditional dual-threshold strategy.
enable_snr_weighting=false
# SNR smoothing parameter controlling weight smoothness (0.0-1.0) - reduced for clearer separation
snr_smoothing=0.02
# Temporal decay factor controlling timestep influence (0.0-1.0) - slightly reduced for more time sensitivity
temporal_decay=0.95
# Minimum weight value to avoid overly small weights (0.0-1.0) - increased to keep important filtered regions
min_weight=0.7
# Maximum weight value to avoid overly large weights (1.0-10.0) - reduced to avoid over-amplification
max_weight=1.2
# Reconstruction consistency weight in SNR (0.0-1.0) - increased to ensure stable detection
consistency_weight=0.8
# Anomaly sensitivity weight in SNR (0.0-1.0) - reduced to balance with consistency
sensitivity_weight=0.2
# Aggregation mode: 'weighted_mean', 'weighted_sum', or 'robust_weighted'
aggregation_mode="robust_weighted"

# --- [Step 4] ---
# Loop runs (useful for ablation experiments)
for round in 1
do
    for w in 2  # For BraTS, w=2 is commonly used; edit here for ablations
    do
        # Set log output directory
        export OPENAI_LOGDIR="${LOG_ROOT}/logs_brats_aba/translation_fpdm_dilated_${w}_${model_num}_${forward_steps}_${round}_x1"
        # Ensure directory exists
        mkdir -p $OPENAI_LOGDIR
        echo "Logs will be saved to: $OPENAI_LOGDIR"

        # Set data directory and pretrained model directory
        data_dir="${DATA_ROOT}/BraTS21_training/preprocessed_data_all_00_128" # Assumes data is under DATA_ROOT
        model_dir="${LOG_ROOT}/logs_brats_normal_99_11_128/logs_guided_${threshold}_all_00_dilated_128_norm"  # Use dilated model
        image_dir="$OPENAI_LOGDIR"

        # Check whether the model directory exists
        if [ ! -d "$model_dir" ]; then
            echo "Error: pretrained model directory not found: $model_dir"
            echo "Make sure the guided model has been trained, or update model_dir to the correct path."
            exit 1
        fi

        # Flags passed to the Python script
        MODEL_FLAGS="--image_size $image_size --num_classes $num_classes --in_channels $in_channels  \
                        --w $w --attention_resolutions 32,16,8 \
                        --num_channels $num_channels --model_num $model_num --ema True\
                        --forward_steps $forward_steps --d_reverse $d_reverse --unet_ver $version"

        DATA_FLAGS="--batch_size 10 --num_batches 1 \
                    --batch_size_val 10 --num_batches_val 10\
                    --modality 0 3 --use_weighted_sampler False --seed $seed"

        DIFFUSION_FLAGS="--null True \
                            --dynamic_clip False \
                            --diffusion_steps $diffusion_steps \
                            --noise_schedule linear \
                            --rescale_learned_sigmas False --rescale_timesteps False"

        DIR_FLAGS="--save_data False --data_dir $data_dir  --image_dir $image_dir --model_dir $model_dir"

        ABLATION_FLAGS="--last_only False --subset_interval -1 --t_e_ratio 1 --use_gradient_sam False --use_gradient_para_sam False"
        
        # Dual-threshold flags
        DUAL_THRESHOLD_FLAGS=""
        if [ "$enable_dual_threshold" = "true" ]; then
            DUAL_THRESHOLD_FLAGS="--enable_dual_threshold --low_quant_offset $low_quant_offset --high_quant_offset $high_quant_offset --entropy_weight $entropy_weight --entropy_threshold $entropy_threshold"
        fi
        
        # SNR-weighting flags
        SNR_WEIGHTING_FLAGS=""
        if [ "$enable_snr_weighting" = "true" ]; then
            SNR_WEIGHTING_FLAGS="--enable_snr_weighting --snr_smoothing $snr_smoothing --temporal_decay $temporal_decay --min_weight $min_weight --max_weight $max_weight --consistency_weight $consistency_weight --sensitivity_weight $sensitivity_weight --aggregation_mode $aggregation_mode"
        fi

        # --- [Step 5] ---
        # Run the image translation script
        NUM_GPUS=1 # Number of GPUs to use
        torchrun --nproc-per-node $NUM_GPUS \
                    --nnodes=1\
                    --rdzv-backend=c10d\
                    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
                ./scripts/translation_dilated.py --name brats $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS $ABLATION_FLAGS $DUAL_THRESHOLD_FLAGS $SNR_WEIGHTING_FLAGS
    done
done

echo "Script completed."
