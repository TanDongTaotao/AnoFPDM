#!/bin/bash

#SBATCH --job-name='clf'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH -p general                
#SBATCH -q public
            
#SBATCH -t 00-10:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


num_classes=2
image_size=128
version=v1
in_channels=1


DATA_ROOT="./data"
LOG_ROOT="./logs"

export MASTER_ADDR=localhost
export MASTER_PORT=12365

export OPENAI_LOGDIR="${LOG_ROOT}/logs_atlas_clf"
mkdir -p "$OPENAI_LOGDIR"

image_dir="$OPENAI_LOGDIR/images"
mkdir -p "$image_dir"

data_dir="${DATA_ROOT}/ATLAS_2/preprocessed_data_t1_00_128"


CLASSIFIER_FLAGS="--unet_ver $version --image_size $image_size --classifier_attention_resolutions 32,16,8 \
                --in_channels $in_channels --out_channels $num_classes \
                --classifier_depth 2 --classifier_width 128 \
                --classifier_pool attention \
                --classifier_resblock_updown True --classifier_use_scale_shift_norm True\
                --data_dir $data_dir --image_dir $image_dir --batch_size 64 --iterations 250000"

NUM_GPUS=1
torchrun --nproc-per-node $NUM_GPUS \
         --nnodes=1\
         --rdzv-backend=c10d\
         --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
        ./scripts/train_classifier.py --name atlas --save_interval 5000\
                                    $CLASSIFIER_FLAGS
