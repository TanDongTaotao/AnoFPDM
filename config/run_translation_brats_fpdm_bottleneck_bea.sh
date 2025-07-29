#!/bin/bash

# 此脚本已修改为可在本地计算机上运行Bottleneck BEA UNet推理，而非 SLURM 集群。
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
export MASTER_PORT=12360 # 使用一个不同的端口以避免冲突

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
version=bottleneck_bea  # Use Bottleneck BEA UNet version

d_reverse=True # 设置为 True 使用 ddim reverse (确定性编码)
               # 否则将使用 ddpm reverse (随机编码)

# --- [步骤 4] ---
# 循环运行，可用于进行消融实验
for round in 1
do
    for w in 2  # 为 brats 数据集选择 w=2; 可在此处修改以进行消融研究
    do
        # 设置日志输出目录
        export OPENAI_LOGDIR="${LOG_ROOT}/logs_brats_aba_bottleneck_bea/translation_fpdm_bottleneck_bea_${w}_${model_num}_${forward_steps}_${round}_x1"
        # 确保目录存在
        mkdir -p $OPENAI_LOGDIR
        echo "日志将保存在: $OPENAI_LOGDIR"

        # 设置数据目录和预训练模型目录
        data_dir="${DATA_ROOT}/BraTS21_training/preprocessed_data_all_00_128" # 假设数据在 DATA_ROOT 下
        model_dir="${LOG_ROOT}/clf_free_guided_bottleneck_bea" # 假设模型在 LOG_ROOT 下
        
        # 检查模型目录是否存在
        if [ ! -d "$model_dir" ]; then
            echo "错误: 模型目录 $model_dir 不存在!"
            echo "请确保您已经训练了模型，或者修改 model_dir 路径。"
            exit 1
        fi
        
        # 检查数据目录是否存在
        if [ ! -d "$data_dir" ]; then
            echo "错误: 数据目录 $data_dir 不存在!"
            echo "请确保您已经预处理了数据，或者修改 data_dir 路径。"
            exit 1
        fi
        
        echo "使用数据目录: $data_dir"
        echo "使用模型目录: $model_dir"
        echo "使用权重 w=$w, 前向步数=$forward_steps, 模型编号=$model_num"
        
        # 运行推理脚本
        python ./scripts/translation_FPDM_bottleneck_bea.py \
            --name brats \
            --data_dir "$data_dir" \
            --model_dir "$model_dir" \
            --batch_size 14 \
            --num_batches 10 \
            --forward_steps $forward_steps \
            --model_num $model_num \
            --ema False \
            --null False \
            --save_data False \
            --num_batches_val 0 \
            --batch_size_val 100 \
            --d_reverse $d_reverse \
            --median_filter True \
            --dynamic_clip False \
            --last_only False \
            --subset_interval -1 \
            --seed $seed \
            --use_weighted_sampler False \
            --use_gradient_sam False \
            --use_gradient_para_sam False \
            --modality 0 3 \
            --t_e_ratio 1 \
            --w $w
        
        echo "推理完成，结果保存在: $OPENAI_LOGDIR"
    done
done

echo "所有推理任务完成!"