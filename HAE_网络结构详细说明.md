# HAE (Heterogeneous AutoEncoder) 家族网络结构详细说明

## 概述

HAE (异构自动编码器) 家族是基于扩散模型的创新网络架构，通过在编码器和解码器中使用不同的网络结构来实现更好的特征表示和生成质量。该家族包含三个主要版本：HAE原版、HAE Lite和HAE V2。

## 核心设计理念

### 异构设计
- **编码器**：采用传统CNN架构，专注于局部特征提取和空间信息压缩
- **解码器**：采用混合CNN-Transformer架构，结合局部建模和长距离依赖建模
- **中间块**：保持标准UNet设计，确保特征传递的稳定性

### 关键创新点
1. **多尺度稀疏Transformer块 (MSTB)**：处理不同尺度的空间信息
2. **混合CNN-Transformer块**：融合CNN的局部感受野和Transformer的全局建模能力
3. **稀疏注意力机制**：减少90%的计算量，提高效率
4. **渐进式优化**：从原版到Lite再到V2，逐步优化参数量和计算效率

---

## 1. HAE 原版 (HAEUNetModel)

### 网络架构特点
- **编码器**：标准CNN架构，使用ResBlock和AttentionBlock
- **解码器**：混合CNN-Transformer架构，在每个上采样前添加异构块
- **参数量**：最大，性能最优
- **适用场景**：高质量生成任务，计算资源充足的环境

### 核心组件

#### 1.1 MultiScaleSparseTransformerBlock (MSTB)
```python
class MultiScaleSparseTransformerBlock(nn.Module):
    def __init__(self, channels, num_heads=8, dropout=0.1, patch_sizes=[1, 4, 8]):
        # 局部信息处理
        self.local_norm = nn.LayerNorm(channels)
        
        # 区域信息处理 - 多尺度
        self.regional_norms = nn.ModuleList([...])
        self.patch_projections = nn.ModuleList([...])
        
        # 稀疏多头注意力
        self.multihead_attn = nn.MultiheadAttention(...)
        
        # 输出投影
        self.mlp = nn.Sequential(...)
```

**特点**：
- 支持多尺度patch处理 (1x1, 4x4, 8x8)
- 为每个尺度创建独立的线性投影层
- 使用完整的4倍扩展MLP
- 稀疏注意力掩码减少90%计算量

#### 1.2 HybridCNNTransformerBlock
```python
class HybridCNNTransformerBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout=0.1, use_checkpoint=False):
        # CNN分支
        self.conv_branch = nn.Sequential(...)
        
        # Transformer分支
        self.transformer_branch = MultiScaleSparseTransformerBlock(...)
        
        # 特征融合
        self.fusion_conv = conv_nd(2, channels * 2, channels, 1)
```

**特点**：
- 双分支并行处理：CNN分支 + Transformer分支
- 通过拼接进行特征融合
- 完整的时间步嵌入处理

### 网络结构图

```
输入 (B, C, H, W)
    |
    v
┌─────────────────┐
│   编码器 (CNN)   │
│  ┌─────────────┐ │
│  │  ResBlock   │ │
│  │     +       │ │
│  │ AttentionBlock│ │
│  └─────────────┘ │
│       ↓         │
│  ┌─────────────┐ │
│  │ Downsample  │ │
│  └─────────────┘ │
└─────────────────┘
         |
         v
┌─────────────────┐
│    中间块       │
│ ResBlock + Attn │
└─────────────────┘
         |
         v
┌─────────────────┐
│ 解码器(混合架构) │
│  ┌─────────────┐ │
│  │ ResBlock +  │ │
│  │ AttentionBlock│ │
│  └─────────────┘ │
│       ↓         │
│  ┌─────────────┐ │
│  │ 混合CNN-    │ │
│  │ Transformer │ │
│  │    块       │ │
│  └─────────────┘ │
│       ↓         │
│  ┌─────────────┐ │
│  │  Upsample   │ │
│  └─────────────┘ │
└─────────────────┘
         |
         v
    输出 (B, C, H, W)
```

---

## 2. HAE Lite (HAEUNetModelLite)

### 网络架构特点
- **编码器**：与原版相同的CNN架构
- **解码器**：优化的混合CNN-Transformer架构
- **参数量**：中等，平衡性能和效率
- **适用场景**：资源受限环境，需要平衡性能和效率

### 核心优化

#### 2.1 MultiScaleSparseTransformerBlockLite
```python
class MultiScaleSparseTransformerBlockLite(nn.Module):
    def __init__(self, channels, num_heads=8, dropout=0.1, patch_sizes=[1, 4, 8]):
        # 共享的卷积投影层 - 关键优化
        self.shared_patch_embed = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )
```

**关键优化**：
- 使用共享卷积投影层替代独立线性投影层
- 借鉴ViT的设计理念
- 大幅减少参数量，保持性能

#### 2.2 HybridCNNTransformerBlockLite
```python
class HybridCNNTransformerBlockLite(TimestepBlock):
    def _forward(self, x, emb):
        # 特征融合：加权平均而不是拼接
        fused = (conv_out + trans_out) / 2
        output = self.fusion_conv(fused)
```

**关键优化**：
- 使用加权平均替代特征拼接
- 减少融合卷积的输入通道数
- 保持相同的表达能力

### 参数优化对比

| 组件 | HAE原版 | HAE Lite | 优化方式 |
|------|---------|----------|----------|
| 多尺度投影 | 独立Linear层 | 共享Conv1x1 | 参数共享 |
| 特征融合 | 拼接+Conv | 平均+Conv | 减少通道数 |
| MLP结构 | 4倍扩展 | 4倍扩展 | 保持不变 |

---

## 3. HAE V2 (HAEUNetModelV2)

### 网络架构特点
- **编码器**：与前两版相同的CNN架构
- **解码器**：进一步优化的混合CNN-Transformer架构
- **参数量**：最小，效率最高
- **适用场景**：移动设备、边缘计算、实时应用

### 核心创新

#### 3.1 BottleneckMLP
```python
class BottleneckMLP(nn.Module):
    def __init__(self, channels, bottleneck_ratio=0.25, dropout=0.1):
        bottleneck_dim = max(1, int(channels * bottleneck_ratio))
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, bottleneck_dim),  # 降维
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, channels),  # 升维
            nn.Dropout(dropout)
        )
```

**创新点**：
- 瓶颈结构：channels → bottleneck_dim → channels
- 默认瓶颈比例0.25，参数量减少约75%
- 保持非线性表达能力

#### 3.2 MultiScaleSparseTransformerBlockV2
```python
class MultiScaleSparseTransformerBlockV2(nn.Module):
    def forward(self, x):
        # 使用共享的卷积投影层处理输入特征
        x_projected = self.shared_patch_embed(x)
        
        # 对投影后的特征进行块划分
        # 平均池化来聚合patch内的信息，而不是展平后线性投影
        x_patches = x_patches.mean(dim=(4, 5))
```

**关键优化**：
- 继承Lite版本的共享投影层
- 使用平均池化替代线性投影
- 集成瓶颈MLP结构
- 进一步减少计算复杂度

### 参数量对比分析

| 版本 | MLP结构 | 投影方式 | 融合方式 | 相对参数量 |
|------|---------|----------|----------|------------|
| HAE原版 | 4倍扩展 | 独立Linear | 拼接 | 100% |
| HAE Lite | 4倍扩展 | 共享Conv1x1 | 平均 | ~70% |
| HAE V2 | 瓶颈0.25倍 | 共享Conv1x1+池化 | 平均 | ~40% |

---

## 技术细节对比

### 稀疏注意力机制
所有版本都使用相同的稀疏注意力策略：

```python
def create_sparse_mask(self, seq_len, sparsity_ratio=0.9):
    mask = th.ones(seq_len, seq_len, dtype=th.bool)
    keep_indices = max(1, int(seq_len * (1 - sparsity_ratio)))
    
    for i in range(seq_len):
        # 保留对角线附近
        start = max(0, i - keep_indices // 2)
        end = min(seq_len, i + keep_indices // 2 + 1)
        mask[i, start:end] = False
        
        # 随机保留一些远距离连接
        if keep_indices > 4:
            num_random = max(1, keep_indices // 4)
            random_indices = th.randperm(seq_len)[:num_random]
            mask[i, random_indices] = False
    
    return mask
```

### 多尺度处理策略

| 尺度 | Patch大小 | 处理方式 | 作用 |
|------|-----------|----------|------|
| 局部 | 1x1 | 直接展平 | 像素级细节 |
| 区域 | 4x4 | 块划分+投影 | 局部模式 |
| 全局 | 8x8 | 块划分+投影 | 全局结构 |

### 时间步嵌入处理
所有版本都支持完整的时间步嵌入和条件嵌入：

```python
# 时间步嵌入
emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

# 条件嵌入（如果有类别条件）
if self.num_classes is not None:
    cemb = self.class_emb(self.label_emb(y))
    emb = emb + cemb
```

---

## 性能特点总结

### HAE原版
- **优势**：最高的生成质量，完整的特征表达能力
- **劣势**：参数量大，计算开销高
- **适用**：高端GPU，追求最佳质量的应用

### HAE Lite
- **优势**：平衡的性能和效率，适中的参数量
- **劣势**：相比原版有轻微的性能损失
- **适用**：中端GPU，平衡性能和资源的应用

### HAE V2
- **优势**：最小的参数量，最高的推理效率
- **劣势**：相比前两版有一定的性能权衡
- **适用**：移动设备，边缘计算，实时应用

---

## 使用建议

1. **高质量生成**：选择HAE原版
2. **平衡应用**：选择HAE Lite
3. **效率优先**：选择HAE V2
4. **渐进式部署**：从V2开始，根据需要升级到Lite或原版

所有版本都保持相同的API接口，可以无缝切换，便于在不同场景下灵活部署。