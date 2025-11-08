#!/bin/bash

# 原始 SLURM 提交参数（仅保留为注释，已翻译，便于参考）：
#SBATCH --job-name='trans_ano'       # 作业名
#SBATCH --nodes=1                    # 使用节点数
#SBATCH --ntasks-per-node=1          # 每节点任务数
#SBATCH --cpus-per-task=4            # 每任务 CPU 核心数
#SBATCH --gres=gpu:a100:1            # GPU 资源配置
#SBATCH --mem=16G                    # 内存上限
#SBATCH -p general                   # 分区
#SBATCH -q public                    # 队列
#SBATCH -t 00-1:00:00                # 最大运行时间
#SBATCH -e ./slurm_out/slurm.%j.err  # 错误输出路径
#SBATCH -o ./slurm_out/slurm.%j.out  # 标准输出路径

# 原脚本环境准备（SLURM 模块命令，保留为注释，已翻译）：
# module purge                       # 清除已加载模块
# module load mamba/latest           # 加载 mamba 模块
# source activate torch_base         # 激活集群上的 conda/mamba 环境

# 此脚本已改写为本地单显卡环境运行（非 SLURM 集群）。
# 运行此脚本前，请确保已激活本地 conda 环境，例如：
# conda activate your_env_name

# --- [步骤 1] ---
# 设置数据和模型目录的根路径（本地路径，根据需要修改）
DATA_ROOT="./data"  # <-- [请修改] 指向您的数据集路径
LOG_ROOT="./logs"   # <-- [请修改] 指向您存放日志和模型的主目录

# --- [步骤 2] ---
# 分布式参数（单机单卡即可）
# [SLURM 原始设置，保留为注释]
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)  # 从节点列表获取主节点地址
# export MASTER_ADDR=$master_addr
# echo $MASTER_ADDR
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))     # 基于作业 ID 计算端口
# echo $MASTER_PORT

# [本地设置]
export MASTER_ADDR=localhost
export MASTER_PORT=12358  # 使用一个避免冲突的端口

# --- [步骤 3] ---
# 模型与推理相关超参数（保留原注释并翻译）
image_size=128             # 图像尺寸
num_classes=0              # 类别数（无监督异常检测设为 0）
in_channels=4              # 输入通道数（BraTS 4 模态）
num_channels=128           # UNet 基础通道数
diffusion_steps=1000       # 扩散步数
seed=0                     # 用于数据加载器与推理复现（统一设置随机种子）

# 保留原参数参考：
# w=-1  # 无引导（unguided）

# 噪声与采样设置（保留原注释并翻译）
# noise_type=gaussian                          # 高斯噪声
# # sample_steps=300                           # 高斯噪声的采样步数
# model_num=250000                             # 训练步数对应的模型
# use_ddpm=True                                # 使用 DDPM 或 DDIM 采样

noise_type=simplex                             # Simplex 噪声
# sample_steps=200                             # Simplex 噪声的采样步数（保持为 200）
model_num=350000                               # 训练步数对应的模型
use_ddpm=True                                  # 使用 DDPM 采样

if [ $use_ddpm = "True" ]
then
    model_name="anoddpm"                      # 使用 DDPM 的模型名
else
    model_name="anoddim"                      # 使用 DDIM 的模型名
fi

# --- [步骤 4] ---
# 循环运行（可用于消融实验，保留原语义）
for round in 1
do
    for sample_steps in 200
    do
        # 设置日志输出目录
        export OPENAI_LOGDIR="${LOG_ROOT}/logs_brats/translation_${model_name}_${noise_type}_${sample_steps}_${round}_plot"
        # 确保目录存在
        mkdir -p "$OPENAI_LOGDIR"
        echo "日志将保存在: $OPENAI_LOGDIR"

        # 设置数据目录和预训练模型目录（请按需修改）
        data_dir="${DATA_ROOT}/BraTS21_training/preprocessed_data_all_00_128"
        model_dir="${LOG_ROOT}/logs_brats_normal_99_11_128/logs_unguided_all_00_v1_128_anoddpm_${noise_type}"
        image_dir="$OPENAI_LOGDIR"

        # 检查模型目录是否存在
        if [ ! -d "$model_dir" ]; then
            echo "错误: 预训练模型目录不存在: $model_dir"
            echo "请确保您已经训练了模型，或者修改 model_dir 变量指向正确的路径。"
            exit 1
        fi

        # 组装传递给 python 脚本的参数
        MODEL_FLAGS="--unet_ver v1 --image_size $image_size --num_classes $num_classes \
                        --in_channels $in_channels --clf_free False \
                        --dropout 0 --attention_resolutions 32,16,8 \
                        --num_channels $num_channels --model_num $model_num --ema True \
                        --use_ddpm $use_ddpm --noise_type $noise_type"

        DATA_FLAGS="--batch_size 100 --num_batches 2 \
                    --batch_size_val 100 --num_batches_val 0 \
                    --modality 0 3 --seed $seed --use_weighted_sampler False"

        DIFFUSION_FLAGS="--diffusion_steps $diffusion_steps \
                            --sample_steps $sample_steps \
                            --noise_schedule linear \
                            --rescale_learned_sigmas False \
                            --rescale_timesteps False \
                            --dynamic_clip False"

        DIR_FLAGS="--save_data True --data_dir $data_dir \
                    --image_dir $image_dir --model_dir $model_dir"

        # --- [步骤 5] ---
        # 运行推理脚本（单显卡）
        NUM_GPUS=1
        torchrun --nproc-per-node $NUM_GPUS \
                    --nnodes=1 \
                    --rdzv-backend=c10d \
                    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
                ./scripts/translation_HEALTHY.py $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS
    done
done

echo "脚本执行完成。"