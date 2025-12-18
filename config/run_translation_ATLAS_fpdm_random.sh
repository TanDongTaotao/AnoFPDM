#!/bin/bash

#SBATCH --job-name='translation_test'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH -p general                
#SBATCH -q grp_twu02
            
#SBATCH -t 01-15:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


DATA_ROOT="./data"
LOG_ROOT="./logs"

export MASTER_ADDR=localhost
export MASTER_PORT=12363

threshold=0.1
num_classes=2
seed=0 # for data loader only
in_channels=1
num_channels=128
image_size=128
forward_steps=600
diffusion_steps=1000
model_num=290000
version=v2

w=30
d_reverse=False # set d_reverse to True for ddim reverse 

for round in 1 2 3
do
  
    export OPENAI_LOGDIR="${LOG_ROOT}/logs_atlas/translation_fpdm_ddpm_${w}_${model_num}_${forward_steps}_${round}_last"
    mkdir -p "$OPENAI_LOGDIR"

    data_dir="${DATA_ROOT}/ATLAS_2/preprocessed_data_t1_00_128"
    model_dir="${LOG_ROOT}/logs_atlas_normal_99_11_128/logs_guided_${threshold}_${version}_t1"
    image_dir="$OPENAI_LOGDIR"

    if [ ! -d "$model_dir" ]; then
        echo "错误: 预训练模型目录不存在: $model_dir"
        exit 1
    fi

    MODEL_FLAGS="--image_size $image_size --num_classes $num_classes --in_channels $in_channels  \
                    --w $w  --dropout 0.1 --attention_resolutions 32,16,8 \
                    --num_channels $num_channels --model_num $model_num --ema True\
                    --forward_steps $forward_steps --d_reverse $d_reverse" 

    DATA_FLAGS="--batch_size 100 --num_batches 40 \
                --batch_size_val 100 --num_batches_val 10\
                --modality 0 --use_weighted_sampler False --seed $seed"

    DIFFUSION_FLAGS="--null True --diffusion_steps $diffusion_steps \
                        --noise_schedule linear \
                        --rescale_learned_sigmas False --rescale_timesteps False"

    DIR_FLAGS="--save_data False --data_dir $data_dir  --image_dir $image_dir --model_dir $model_dir"
    ABLATION_FLAGS="--last_only True --subset_interval -1 --t_e_ratio 1 --use_gradient_sam False"

    NUM_GPUS=1
    torchrun --nproc-per-node $NUM_GPUS \
                --nnodes=1\
                --rdzv-backend=c10d\
                --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
            ./scripts/translation_FPDM.py --name atlas $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS $ABLATION_FLAGS
done
