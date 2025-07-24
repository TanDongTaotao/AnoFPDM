#!/bin/bash

# 此脚本已修改为可在本地计算机上运行BEA UNet推理，而非 SLURM 集群。
# 运行此脚本前，请确保您已激活 conda 环境。
# 例如: `conda activate your_env_name`
# 或者取消下面的注释:
# source activate torch_base

# --- [步骤 1] ---
# 设置数据和模型目录的根路径
# 请根据您的实际情况修改这些路径
DATA_ROOT="./data" # <-- [请修改] 指向您的数据集路径
LOG_ROOT="./logs"  # <-- [请修改] 指向您存放日志和模型的主目录

# --- [步骤 2] ---
# 为分布式训练设置网络参数
# 如果您在单机上运行，可以保持默认设置
export MASTER_ADDR=localhost
export MASTER_PORT=12359 # 使用一个不同的端口以避免冲突

# --- [步骤 3] ---
# 设置模型和训练相关的超参数
threshold=0.1
num_classes=2
seed=0 # 仅用于数据加载器
in_channels=4
num_channels=128
image_size=128
forward_steps=600
diffusion_steps=1000
model_num=210000
version=bea  # Use BEA UNet version

d_reverse=True # 设置为 True 使用 ddim reverse (确定性编码)
               # 否则将使用 ddpm reverse (随机编码)

# --- [步骤 4] ---
# 循环运行，可用于进行消融实验
for round in 1
do
    for w in 2  # 为 brats 数据集选择 w=2; 可在此处修改以进行消融研究
    do
        # 设置日志输出目录
        export OPENAI_LOGDIR="${LOG_ROOT}/logs_brats_aba_bea/translation_fpdm_bea_${w}_${model_num}_${forward_steps}_${round}_x1"
        # 确保目录存在
        mkdir -p $OPENAI_LOGDIR
        echo "日志将保存在: $OPENAI_LOGDIR"

        # 设置数据目录和预训练模型目录
        data_dir="${DATA_ROOT}/BraTS21_training/preprocessed_data_all_00_128" # 假设数据在 DATA_ROOT 下
        model_dir="${LOG_ROOT}/logs_brats_normal_99_11_128_bea/logs_guided_${threshold}_all_00_${version}_128_norm"
        image_dir="$OPENAI_LOGDIR"

        # 检查模型目录是否存在
        if [ ! -d "$model_dir" ]; then
            echo "错误: 预训练BEA模型目录不存在: $model_dir"
            echo "请确保您已经训练了BEA引导模型，或者修改 model_dir 变量指向正确的路径。"
            exit 1
        fi

        # 设置传递给 python 脚本的参数
        MODEL_FLAGS="--image_size $image_size --num_classes $num_classes --in_channels $in_channels  \
                        --w $w --attention_resolutions 32,16,8 \
                        --num_channels $num_channels --model_num $model_num --ema True\
                        --forward_steps $forward_steps --d_reverse $d_reverse --unet_ver $version --use_bea True"

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

        # --- [步骤 5] ---
        # 运行BEA UNet图像翻译脚本
        NUM_GPUS=1 # 设置使用的 GPU 数量
        torchrun --nproc-per-node $NUM_GPUS \
                    --nnodes=1\
                    --rdzv-backend=c10d\
                    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
                ./scripts/translation_FPDM_bea.py --name brats $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS $ABLATION_FLAGS
    done
done

echo "BEA UNet推理脚本执行完成。"