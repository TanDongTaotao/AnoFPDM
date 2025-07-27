#!/bin/bash

# 此脚本已修改为可在本地计算机上运行HAE UNet推理，而非 SLURM 集群。
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
export MASTER_PORT=12361 # 使用一个不同的端口以避免冲突

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
version=hae  # Use HAE UNet version

d_reverse=True # 设置为 True 使用 ddim reverse (确定性编码)
               # 否则将使用 ddpm reverse (随机编码)

# --- [步骤 4] ---
# 循环运行，可用于进行消融实验
for round in 1
do
    for w in 2  # 为 brats 数据集选择 w=2; 可在此处修改以进行消融研究
    do
        # 设置日志输出目录
        export OPENAI_LOGDIR="$LOG_ROOT/clf_free_guided_hae"
        
        # 设置模型目录
        model_dir="$OPENAI_LOGDIR"
        
        # 设置结果输出目录
        image_dir="$OPENAI_LOGDIR/results_hae_w${w}_fs${forward_steps}_r${round}"
        
        # 数据标志
        DATA_FLAGS="--data_dir $DATA_ROOT --image_size $image_size --num_classes $num_classes --class_cond True --ret_lab True --mixed True"
        
        # 模型标志
        MODEL_FLAGS="--unet_ver $version --clf_free True --use_hae True \
                     --in_channels $in_channels \
                     --num_channels $num_channels \
                     --attention_resolutions 32,16,8 \
                     --learn_sigma False \
                     --dropout 0.1"
        
        # 扩散标志
        DIFFUSION_FLAGS="--diffusion_steps $diffusion_steps \
                         --noise_schedule linear \
                         --rescale_learned_sigmas False \
                         --rescale_timesteps False"
        
        # 推理标志
        INFERENCE_FLAGS="--model_dir $model_dir \
                         --model_num $model_num \
                         --image_dir $image_dir \
                         --w $w \
                         --forward_steps $forward_steps \
                         --d_reverse $d_reverse \
                         --seed $seed \
                         --batch_size 1 \
                         --num_batches 10 \
                         --modality FLAIR \
                         --median_filter True \
                         --t_e_ratio 0.5 \
                         --ema True"
        
        echo "开始 HAE UNet 推理，轮次: $round, w: $w"
        echo "模型目录: $model_dir"
        echo "结果目录: $image_dir"
        
        # 运行推理
        python ./scripts/translation_FPDM_hae.py --name brats \
                                                  $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $INFERENCE_FLAGS
        
        echo "HAE UNet 推理完成，轮次: $round, w: $w"
        echo "结果已保存到: $image_dir"
        echo "-----------------------------------"
    done
done

echo "所有 HAE UNet 推理任务完成！"