#!/bin/bash

# 说明：原脚本使用 SLURM 集群调度。为适配本地 Linux 单卡服务器，现改为本地运行方式。
# 保留原语义，并在注释中给出中文翻译与说明。

# 原 SLURM 指令（已禁用，仅保留以供参考）
#SBATCH --job-name='translation_test'        # 任务名（原 SLURM 配置）
#SBATCH --nodes=1                            # 节点数
#SBATCH --ntasks-per-node=1                  # 每节点任务数
#SBATCH --cpus-per-task=4                    # 每任务 CPU 数
#SBATCH --gres=gpu:a100:1                    # GPU 资源（A100 单卡）
#SBATCH --mem=16G                            # 内存
#SBATCH -p general                           # 队列
#SBATCH -q public                            # QoS
#SBATCH -t 01-00:00:00                       # 超时
#SBATCH -e ./slurm_out/slurm.%j.err          # 错误日志
#SBATCH -o ./slurm_out/slurm.%j.out          # 输出日志

# --- 环境设置（本地单机单卡） ---
# 如果使用 conda，请先激活环境。例如：
# conda activate torch_base

# 为分布式初始化设置地址与端口（单机默认即可）
export MASTER_ADDR=localhost
export MASTER_PORT=12361

# 可选：限制使用的 GPU（按需设置）
# export CUDA_VISIBLE_DEVICES=0

# --- 路径与超参数 ---
DATA_ROOT="./data"       # 请修改为实际数据集根目录
LOG_ROOT="./logs"        # 请修改为模型与日志根目录

threshold=0.1             # 原注释：for model trained with p=0.1
image_size=128
num_classes=2
in_channels=4
num_channels=128
seed=0                    # 原注释：for data loader only（仅影响数据加载器随机数）

diffusion_steps=1000
version=v2                # 使用 v2 UNet（与原脚本一致）

# 循环参数：可按需网格搜索
for model_num in 210000; do
  for w in 1.1; do                 # 原注释：you can change this in validation set
    for sample_steps in 450; do    # 原注释：you can change this in validation set

      export OPENAI_LOGDIR="${LOG_ROOT}/logs_brats/translation_ddib_${w}_${sample_steps}_${model_num}"
      mkdir -p "$OPENAI_LOGDIR"
      echo "日志将保存在: $OPENAI_LOGDIR"

      data_dir="${DATA_ROOT}/BraTS21_training/preprocessed_data_all_00_128"
      model_dir="${LOG_ROOT}/logs_brats_normal_99_11_128/logs_guided_${threshold}_all_00_${version}_128_norm"
      image_dir="$OPENAI_LOGDIR"

      # 检查模型目录是否存在
      if [ ! -d "$model_dir" ]; then
        echo "错误: 预训练模型目录不存在: $model_dir"
        echo "请确认已训练好模型，或修改 model_dir 指向正确路径。"
        exit 1
      fi

      # 组装传入 Python 的参数（保持原语义）
      MODEL_FLAGS="--unet_ver $version --image_size $image_size --num_classes $num_classes \
                    --in_channels $in_channels --clf_free True \
                    --w $w --attention_resolutions 32,16,8 \
                    --num_channels $num_channels --model_num $model_num --ema True \
                    --learn_sigma False"

      DATA_FLAGS="--batch_size 100 --num_batches 1 \
                  --batch_size_val 100 --num_batches_val 10 \
                  --modality 0 3 --use_weighted_sampler False --seed $seed"

      DIFFUSION_FLAGS="--diffusion_steps $diffusion_steps \
                        --sample_steps $sample_steps \
                        --noise_schedule linear \
                        --rescale_learned_sigmas False \
                        --rescale_timesteps False \
                        --dynamic_clip False"  # 原注释：seems not working（似乎无效）

      DIR_FLAGS="--save_data False --data_dir $data_dir \
                  --image_dir $image_dir --model_dir $model_dir"

      # 单机单卡运行（torchrun 初始化分布式，world size=1）
      NUM_GPUS=1
      torchrun --nproc-per-node $NUM_GPUS \
               --nnodes=1 \
               --rdzv-backend=c10d \
               --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
               ./scripts/translation_DDIB.py --name brats $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS
    done
  done
done

echo "DDIB BrATS 推理脚本执行完成。"