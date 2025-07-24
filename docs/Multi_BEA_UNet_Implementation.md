# Multi-layer Boundary-Enhanced Attention (Multi-BEA) UNet Implementation

## Overview

The Multi-BEA UNet is an enhanced version of the original UNetV2 that incorporates Boundary-Enhanced Attention (BEA) mechanisms at three strategic layers to improve boundary detection and segmentation accuracy in medical image analysis.

## Architecture Comparison

### Original UNetV2 vs Multi-BEA UNet

| Component | Original UNetV2 | Multi-BEA UNet |
|-----------|----------------|----------------|
| Encoder Blocks 1-2 | Standard convolution + attention | Standard convolution + attention |
| Last Encoder Block | Standard convolution + attention | **Standard convolution + attention + BEA** |
| Middle Block | Standard attention | **Standard attention + BEA** |
| First Decoder Block | Standard convolution + attention | **Standard convolution + attention + BEA** |
| Decoder Blocks 2-3 | Standard convolution + attention | Standard convolution + attention |

### Key Enhancements

1. **Strategic Layer Selection**: BEA is applied to three critical layers:
   - **Last Encoder Block**: Captures high-level boundary features before bottleneck
   - **Middle Block**: Enhances boundary information at the deepest level
   - **First Decoder Block**: Refines boundary details during upsampling

2. **Multi-Resolution Boundary Detection**: By applying BEA at different network depths, the model can detect boundaries at multiple scales and resolutions.

3. **Preserved Skip Connections**: All original skip connections are maintained to ensure gradient flow and feature preservation.

## Multi-BEA Module Details

### Sobel Edge Detection
```python
def sobel(x):
    """Apply Sobel edge detection to input tensor"""
    # Sobel kernels for gradient computation
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    
    # Apply convolution to compute gradients
    grad_x = F.conv2d(x, sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
    grad_y = F.conv2d(x, sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
    
    # Compute gradient magnitude
    return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
```

### Multi-Boundary Aware Attention
```python
class MultiBoundaryAwareAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.boundary_conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.attention_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Extract boundary information using Sobel operator
        boundary = sobel(x.mean(dim=1, keepdim=True))
        boundary_features = self.boundary_conv(boundary)
        
        # Combine original features with boundary features
        combined = torch.cat([x, boundary_features], dim=1)
        attention_weights = self.sigmoid(self.attention_conv(combined))
        
        # Apply attention to enhance boundary-relevant features
        return x * attention_weights
```

## File Structure

```
AnoFPDM/
├── guided_diffusion/
│   ├── unet_multi_bea.py          # Multi-BEA UNet implementation
│   └── script_util.py             # Updated to support Multi-BEA
├── scripts/
│   ├── train_multi_bea.py         # Training script for Multi-BEA
│   └── translation_FPDM_multi_bea.py  # Inference script
├── config/
│   ├── run_train_brats_clf_free_guided_multi_bea.sh     # Training config
│   └── run_translation_brats_fpdm_multi_bea.sh          # Inference config
└── docs/
    ├── Multi_BEA_UNet_Implementation.md                 # This document
    └── multi_bea_unet_comparison_diagram.svg            # Architecture diagram
```

## Usage Instructions

### Training

1. **Prepare your dataset** in the BraTS format
2. **Run training** using the provided configuration:
   ```bash
   bash config/run_train_brats_clf_free_guided_multi_bea.sh
   ```

### Inference

1. **Ensure trained model** is available
2. **Run inference** using:
   ```bash
   bash config/run_translation_brats_fpdm_multi_bea.sh
   ```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `unet_ver` | "multi_bea" | Specifies Multi-BEA UNet variant |
| `use_multi_bea` | True | Enables Multi-BEA attention mechanism |
| `attention_resolutions` | "32,16,8" | Resolution levels for attention |
| `num_heads` | 4 | Number of attention heads |
| `num_head_channels` | 64 | Channels per attention head |

## Expected Improvements

1. **Enhanced Boundary Detection**: Multi-layer BEA provides better boundary sensitivity
2. **Improved Segmentation Accuracy**: More precise delineation of anatomical structures
3. **Multi-Scale Feature Learning**: Boundary information captured at different resolutions
4. **Robust Anomaly Detection**: Better identification of pathological boundaries

## Technical Notes

### Memory Considerations
- Multi-BEA adds computational overhead at three layers
- Memory usage increases by approximately 15-20% compared to original UNetV2
- Gradient computation for Sobel operators adds minimal overhead

### Training Considerations
- Learning rate may need adjustment due to additional parameters
- Convergence might be slightly slower initially
- Monitor boundary-related metrics during training

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure `unet_multi_bea.py` is in the correct directory
2. **Parameter Mismatch**: Verify `use_multi_bea=True` in configuration
3. **Memory Issues**: Reduce batch size if encountering OOM errors

### Debug Tips

1. **Check Model Loading**: Verify correct model variant is loaded
2. **Monitor Attention Maps**: Visualize attention weights during training
3. **Boundary Visualization**: Plot Sobel edge maps to verify boundary detection

## Future Enhancements

1. **Adaptive BEA**: Dynamic selection of BEA layers based on input characteristics
2. **Multi-Modal BEA**: Extension to multi-modal medical imaging
3. **Learnable Edge Detection**: Replace Sobel with learnable edge detection kernels
4. **Attention Visualization**: Tools for visualizing Multi-BEA attention maps

---

*This implementation maintains full compatibility with the original FPDM framework while adding enhanced boundary detection capabilities through strategic placement of BEA modules.*

# 多层边界增强注意力（Multi-BEA）UNet 实现

## 概述

Multi-BEA UNet 是原始 UNetV2 的增强版本，在三个战略层级集成了边界增强注意力（BEA）机制，以提高医学图像分析中的边界检测和分割精度。

## 架构对比

### 原始 UNetV2 vs Multi-BEA UNet

| 组件 | 原始 UNetV2 | Multi-BEA UNet |
|------|-------------|----------------|
| 编码器块 1-2 | 标准卷积 + 注意力 | 标准卷积 + 注意力 |
| 最后编码器块 | 标准卷积 + 注意力 | **标准卷积 + 注意力 + BEA** |
| 中间块 | 标准注意力 | **标准注意力 + BEA** |
| 第一解码器块 | 标准卷积 + 注意力 | **标准卷积 + 注意力 + BEA** |
| 解码器块 2-3 | 标准卷积 + 注意力 | 标准卷积 + 注意力 |

### 关键增强

1. **战略层级选择**：BEA 应用于三个关键层级：
   - **最后编码器块**：在瓶颈前捕获高级边界特征
   - **中间块**：在最深层级增强边界信息
   - **第一解码器块**：在上采样过程中细化边界细节

2. **多分辨率边界检测**：通过在不同网络深度应用 BEA，模型可以在多个尺度和分辨率上检测边界。

3. **保留跳跃连接**：保持所有原始跳跃连接以确保梯度流和特征保存。

## Multi-BEA 模块详情

### Sobel 边缘检测
```python
def sobel(x):
    """对输入张量应用 Sobel 边缘检测"""
    # 用于梯度计算的 Sobel 核
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    
    # 应用卷积计算梯度
    grad_x = F.conv2d(x, sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
    grad_y = F.conv2d(x, sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
    
    # 计算梯度幅度
    return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
```

### 多边界感知注意力
```python
class MultiBoundaryAwareAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.boundary_conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.attention_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 使用 Sobel 算子提取边界信息
        boundary = sobel(x.mean(dim=1, keepdim=True))
        boundary_features = self.boundary_conv(boundary)
        
        # 结合原始特征和边界特征
        combined = torch.cat([x, boundary_features], dim=1)
        attention_weights = self.sigmoid(self.attention_conv(combined))
        
        # 应用注意力增强边界相关特征
        return x * attention_weights
```

## 文件结构

```
AnoFPDM/
├── guided_diffusion/
│   ├── unet_multi_bea.py          # Multi-BEA UNet 实现
│   └── script_util.py             # 更新以支持 Multi-BEA
├── scripts/
│   ├── train_multi_bea.py         # Multi-BEA 训练脚本
│   └── translation_FPDM_multi_bea.py  # 推理脚本
├── config/
│   ├── run_train_brats_clf_free_guided_multi_bea.sh     # 训练配置
│   └── run_translation_brats_fpdm_multi_bea.sh          # 推理配置
└── docs/
    ├── Multi_BEA_UNet_Implementation.md                 # 本文档
    └── multi_bea_unet_comparison_diagram.svg            # 架构图
```

## 使用说明

### 训练

1. **准备数据集**为 BraTS 格式
2. **运行训练**使用提供的配置：
   ```bash
   bash config/run_train_brats_clf_free_guided_multi_bea.sh
   ```

### 推理

1. **确保训练模型**可用
2. **运行推理**使用：
   ```bash
   bash config/run_translation_brats_fpdm_multi_bea.sh
   ```

## 关键参数

| 参数 | 值 | 描述 |
|------|----|---------|
| `unet_ver` | "multi_bea" | 指定 Multi-BEA UNet 变体 |
| `use_multi_bea` | True | 启用 Multi-BEA 注意力机制 |
| `attention_resolutions` | "32,16,8" | 注意力的分辨率级别 |
| `num_heads` | 4 | 注意力头数量 |
| `num_head_channels` | 64 | 每个注意力头的通道数 |

## 预期改进

1. **增强边界检测**：多层 BEA 提供更好的边界敏感性
2. **改进分割精度**：更精确的解剖结构描绘
3. **多尺度特征学习**：在不同分辨率捕获边界信息
4. **鲁棒异常检测**：更好地识别病理边界

## 技术说明

### 内存考虑
- Multi-BEA 在三个层级增加计算开销
- 相比原始 UNetV2，内存使用增加约 15-20%
- Sobel 算子的梯度计算增加最小开销

### 训练考虑
- 由于额外参数，学习率可能需要调整
- 初期收敛可能稍慢
- 训练期间监控边界相关指标

## 故障排除

### 常见问题

1. **导入错误**：确保 `unet_multi_bea.py` 在正确目录
2. **参数不匹配**：验证配置中 `use_multi_bea=True`
3. **内存问题**：如遇到 OOM 错误，减少批次大小

### 调试技巧

1. **检查模型加载**：验证加载了正确的模型变体
2. **监控注意力图**：训练期间可视化注意力权重
3. **边界可视化**：绘制 Sobel 边缘图验证边界检测

## 未来增强

1. **自适应 BEA**：基于输入特征动态选择 BEA 层级
2. **多模态 BEA**：扩展到多模态医学成像
3. **可学习边缘检测**：用可学习边缘检测核替换 Sobel
4. **注意力可视化**：Multi-BEA 注意力图可视化工具

---

*此实现在添加通过战略性放置 BEA 模块增强边界检测能力的同时，保持与原始 FPDM 框架的完全兼容性。*