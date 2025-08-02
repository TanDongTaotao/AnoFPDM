# BEA双层损失实现方案

本文档描述了基于Kimi方案实现的BEA（Boundary-Enhanced Attention）双层损失扩散模型，该方案在原有BEA模型基础上增加了边界感知一致性损失，以提升异常检测性能。

## 方案概述

### 核心思想
单独"插注意力"就像给发动机换了新喷油嘴却还用老ECU程序——**结构有了，但损失函数没给它"任务"**。加入瓶颈层BEA后，建议把**扩散MSE损失**升级为**"双层损失"**：

1. **扩散级**：仍用MSE保证去噪质量
2. **特征级**：用**"边界感知一致性损失"**让BEA真正学到"边界→异常"映射

### 双层损失公式

```python
def total_loss(x_pred, x0, feat_enc, feat_dec, λ=0.3):
    # 1. 扩散去噪损失
    L_diff = F.mse_loss(x_pred, x0)

    # 2. 边界感知一致性损失
    # 计算 Sobel 边界权重
    g = sobel(x0)                           # [B,1,H,W]
    g = F.interpolate(g, size=feat_enc.shape[-2:])
    w = g / (g.mean() + 1e-6)               # 归一化

    # 逐像素特征差异
    diff_feat = torch.abs(feat_enc - feat_dec)
    L_bea = (w * diff_feat).mean()

    return L_diff + λ * L_bea
```

### 预期增益

| 损失组合 | AUPRC | 备注 |
|---|---|---|
| 仅 MSE | 0.722 | 原基线 |
| MSE + 边界一致性 (λ=0.3) | **0.748-0.756** | +2.6-3.4，显存无额外开销 |
| 调 λ 至 0.5 | 0.743 | 过拟合风险增大 |

## 实现文件结构

### 新增核心文件

1. **`guided_diffusion/unet_bea_dual_loss.py`**
   - 基于原BEA UNet模型扩展的双层损失版本
   - 在forward方法中返回编码器和解码器特征
   - 保持与原BEA模型完全一致的结构

2. **`guided_diffusion/gaussian_diffusion_dual_loss.py`**
   - 扩展的高斯扩散类，支持双层损失计算
   - 新增`LossType.DUAL_LOSS`损失类型
   - 实现`boundary_aware_consistency_loss`方法
   - 使用Sobel算子计算边界权重

3. **`guided_diffusion/train_util_dual_loss.py`**
   - 双层损失训练循环类
   - 基于原`TrainLoop`类进行扩展
   - 保持训练流程完全一致

### 训练和推理脚本

4. **`scripts/train_bea_dual_loss.py`**
   - 双层损失训练脚本
   - 基于`train_bea.py`进行修改
   - 新增`boundary_loss_weight`参数控制λ值

5. **`scripts/translation_FPDM_bea_dual_loss.py`**
   - 双层损失推理脚本
   - 基于`translation_FPDM_bea.py`进行修改
   - 保持推理流程和数据保存格式一致

### Shell脚本

6. **`config/run_train_brats_clf_free_guided_bea_dual_loss.sh`**
   - 双层损失训练Shell脚本
   - 默认λ=0.3
   - 使用不同端口避免冲突

7. **`config/run_translation_brats_fpdm_bea_dual_loss.sh`**
   - 双层损失推理Shell脚本
   - 支持批量推理和结果保存

### 修改的现有文件

8. **`guided_diffusion/script_util.py`**
   - 添加对`bea_dual_loss`版本的支持
   - 在`create_model`函数中增加相应分支

## 使用方法

### 训练

```bash
# 激活环境
source activate torch

# 运行训练
bash config/run_train_brats_clf_free_guided_bea_dual_loss.sh
```

### 推理

```bash
# 运行推理
bash config/run_translation_brats_fpdm_bea_dual_loss.sh
```

### 参数调整

- **boundary_loss_weight**: 控制边界感知一致性损失的权重λ，默认0.3
- **use_bea**: 启用BEA模块，默认True
- 其他参数与原BEA模型保持一致

## 技术特点

### 1. 代码一致性
- 所有新文件都基于现有BEA代码进行扩展
- 保持相同的代码风格和结构
- 最小化对现有代码的修改

### 2. 边界感知机制
- 使用Sobel算子计算图像边界
- 动态权重分配，突出边界区域
- 特征级一致性约束

### 3. 零像素标签
- 无需额外的像素级标注
- 仅使用原始图像计算边界权重
- 适用于无监督异常检测

### 4. 显存友好
- 边界计算开销极小
- 无额外显存占用
- 训练效率与原模型相当

## 实验配置

### 默认超参数
- **学习率**: 1e-4
- **批次大小**: 14
- **扩散步数**: 1000
- **边界损失权重**: 0.3
- **图像尺寸**: 128x128
- **输入通道**: 4（BraTS数据集）

### 数据集支持
- BraTS21（脑肿瘤分割）
- ATLAS（脑卒中病变）
- 其他医学图像数据集

## 注意事项

1. **模型兼容性**: 双层损失模型与原BEA模型结构完全一致，可以无缝切换
2. **训练稳定性**: λ值建议在0.1-0.5范围内，过大可能导致训练不稳定
3. **推理性能**: 推理时不计算边界损失，性能与原模型相同
4. **结果保存**: 保持与原BEA脚本相同的数据保存格式，便于对比分析

## 文件依赖关系

```
guided_diffusion/
├── unet_bea_dual_loss.py          # 双层损失UNet模型
├── gaussian_diffusion_dual_loss.py # 双层损失扩散过程
├── train_util_dual_loss.py        # 双层损失训练工具
└── script_util.py                 # 模型创建工具（已修改）

scripts/
├── train_bea_dual_loss.py         # 训练脚本
└── translation_FPDM_bea_dual_loss.py # 推理脚本

config/
├── run_train_brats_clf_free_guided_bea_dual_loss.sh     # 训练Shell
└── run_translation_brats_fpdm_bea_dual_loss.sh          # 推理Shell
```

这个实现方案完全基于现有BEA代码，保持了代码的一致性和兼容性，同时引入了Kimi提出的双层损失机制，预期能够显著提升异常检测性能。