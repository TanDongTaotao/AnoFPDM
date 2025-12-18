#!/bin/bash

#SBATCH --job-name='translation_test'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH -p general                
#SBATCH -q public 
            
#SBATCH -t 01-00:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


threshold=0.1 # for model trained with p=0.1
image_size=128
w_reverse=-1
num_classes=2
in_channels=1
num_channels=128
seed=0 # for data loader only

DATA_ROOT="./data"
LOG_ROOT="./logs"

export MASTER_ADDR=localhost
export MASTER_PORT=12360

# w=1.1
diffusion_steps=1000
version=v2


for model_num in 290000
do
    for w in 1.3  # you can change this to other values in validation set (grid search)
    do
        for sample_steps in 300   # you can change this to other values in validation set (grid search)
        do
            
            export OPENAI_LOGDIR="${LOG_ROOT}/logs_atlas/translation_ddib_${w}_${sample_steps}_${model_num}"
            mkdir -p "$OPENAI_LOGDIR"
            data_dir="${DATA_ROOT}/ATLAS_2/preprocessed_data_t1_00_128"
            model_dir="${LOG_ROOT}/logs_atlas_normal_99_11_128/logs_guided_${threshold}_${version}_t1"
           
            image_dir="$OPENAI_LOGDIR"

            if [ ! -d "$model_dir" ]; then
                echo "错误: 预训练模型目录不存在: $model_dir"
                exit 1
            fi

            MODEL_FLAGS="--unet_ver $version --image_size $image_size --num_classes $num_classes \
                        --in_channels $in_channels --clf_free True\
                            --w $w  --attention_resolutions 32,16,8\
                            --num_channels $num_channels --model_num $model_num --ema True\
                            --learn_sigma False"

            DATA_FLAGS="--batch_size 100 --num_batches 40\
                        --batch_size_val 100 --num_batches_val 0\
                        --modality 0 --use_weighted_sampler False --seed $seed"


            DIFFUSION_FLAGS="--diffusion_steps $diffusion_steps\
                                --sample_steps $sample_steps\
                                --noise_schedule linear\
                                --rescale_learned_sigmas False\
                                --rescale_timesteps False\
                                --dynamic_clip False"

            DIR_FLAGS="--save_data False --data_dir $data_dir\
                        --image_dir $image_dir --model_dir $model_dir"


            NUM_GPUS=1
            torchrun --nproc-per-node $NUM_GPUS\
                    --nnodes=1\
                    --rdzv-backend=c10d\
                    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
                    ./scripts/translation_DDIB.py --name atlas $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS
        done
    done
done
