#!/bin/bash

#SBATCH --job-name='clf_guided'
#SBATCH --nodes=1                    
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH -p general                
#SBATCH -q public
            
#SBATCH -t 03-00:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


module purge
module load mamba/latest
source activate torch_base


in_channels=4
batch_size=32
save_interval=10000
num_classes=2 # healthy or unhealthy
image_size=128
version=v1 # unet version

# log directory
export OPENAI_LOGDIR="./logs/clf_guided"
# data directory
data_dir="/data/preprocessed_data"
# saved images directory, visual check
image_dir="$OPENAI_LOGDIR/images"

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

EVA_FLAGS="--save_interval $save_interval --sample_shape 12 $in_channels $image_size $image_size \
                --timestep_respacing ddim1000"


# slurm setup
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo $MASTER_ADDR

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo $MASTER_PORT

NUM_GPUS=2
torchrun --nproc-per-node $NUM_GPUS \
         --nnodes=1\
         --rdzv-backend=c10d\
         --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
        ./scripts/train.py --name brats --resume_checkpoint $resume_checkpoint\
                            $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $GUI_FLAGS $EVA_FLAGS



