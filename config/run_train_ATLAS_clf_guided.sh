#!/bin/bash

#SBATCH --job-name='clf_guided'
#SBATCH --nodes=1                    
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=16G
#SBATCH -p general                
#SBATCH -q public
            
#SBATCH -t 01-15:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


in_channels=1
batch_size=64
save_interval=5000
num_classes=2 # healthy or unhealthy
image_size=128
version=v1 # unet version

# log directory
DATA_ROOT="./data"
LOG_ROOT="./logs"

export OPENAI_LOGDIR="${LOG_ROOT}/logs_atlas_normal_99_11_128/logs_clf_guided_v1"
mkdir -p "$OPENAI_LOGDIR"

# data directory
data_dir="${DATA_ROOT}/ATLAS_2/preprocessed_data_t1_00_128"
# saved images directory, visual check
image_dir="$OPENAI_LOGDIR/images"
mkdir -p "$image_dir"

DATA_FLAGS="--image_size $image_size --num_classes $num_classes --class_cond True --ret_lab True --mixed True"

MODEL_FLAGS="--unet_ver $version \
             --in_channels $in_channels \
             --num_channels 128 \
             --attention_resolutions 32,16,8\
             --learn_sigma True"

DIFFUSION_FLAGS="--clf_free False\
                 --diffusion_steps 1000\
                    --noise_schedule linear \
                    --rescale_learned_sigmas False \
                    --rescale_timesteps False"

TRAIN_FLAGS="--data_dir $data_dir --image_dir $image_dir --batch_size $batch_size"

EVA_FLAGS="--save_interval $save_interval --sample_shape 12 $in_channels $image_size $image_size" 

GUI_FLAGS=""

export MASTER_ADDR=localhost
export MASTER_PORT=12366

NUM_GPUS=1
torchrun --nproc-per-node $NUM_GPUS \
         --nnodes=1\
         --rdzv-backend=c10d\
         --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
        ./scripts/train.py --name atlas $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $GUI_FLAGS $EVA_FLAGS



