# HAE家族使用指南

## 1. 快速开始

### 1.1 基本导入

```python
from guided_diffusion.unet_hae import HAEUNetModel
from guided_diffusion.unet_hae_lite import HAEUNetModelLite
from guided_diffusion.unet_hae_v2 import HAEUNetModelV2
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
```

### 1.2 基础配置

```python
# 获取默认配置
defaults = model_and_diffusion_defaults()

# HAE通用配置
hae_config = {
    'image_size': 256,
    'in_channels': 3,
    'model_channels': 128,
    'out_channels': 3,
    'num_res_blocks': 2,
    'attention_resolutions': '16,8',
    'dropout': 0.1,
    'channel_mult': '',  # 自动根据image_size设置
    'use_hae': True,
    'num_heads': 8,
    'use_checkpoint': False,
    'use_fp16': False,
}

# 更新默认配置
defaults.update(hae_config)
```

## 2. 各版本详细使用

### 2.1 HAE原版 - 高质量生成

#### 创建模型

```python
def create_hae_original(image_size=256, model_channels=128):
    """创建HAE原版模型"""
    model = HAEUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=model_channels,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        use_hae=True,
        num_heads=8,
        num_head_channels=64,
        use_checkpoint=False,
        use_fp16=False,
        clf_free=True,
    )
    return model

# 使用示例
model = create_hae_original()
print(f"HAE原版参数量: {sum(p.numel() for p in model.parameters()):,}")
```

#### 训练配置

```python
# 训练超参数（HAE原版）
training_config = {
    'lr': 1e-4,
    'batch_size': 16,  # 根据GPU内存调整
    'weight_decay': 0.01,
    'ema_rate': 0.9999,
    'log_interval': 100,
    'save_interval': 10000,
    'microbatch': 4,  # 梯度累积
}
```

### 2.2 HAE Lite - 平衡性能

#### 创建模型

```python
def create_hae_lite(image_size=256, model_channels=128):
    """创建HAE Lite模型"""
    model = HAEUNetModelLite(
        image_size=image_size,
        in_channels=3,
        model_channels=model_channels,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        use_hae=True,
        num_heads=8,
        num_head_channels=64,
        use_checkpoint=True,  # 启用检查点节省内存
        use_fp16=False,
        clf_free=True,
    )
    return model

# 使用示例
model = create_hae_lite()
print(f"HAE Lite参数量: {sum(p.numel() for p in model.parameters()):,}")
```

#### 训练配置

```python
# 训练超参数（HAE Lite）
training_config = {
    'lr': 1.5e-4,  # 稍高的学习率
    'batch_size': 24,  # 更大的批次
    'weight_decay': 0.01,
    'ema_rate': 0.9999,
    'log_interval': 100,
    'save_interval': 10000,
    'microbatch': 6,
}
```

### 2.3 HAE V2 - 高效率部署

#### 创建模型

```python
def create_hae_v2(image_size=256, model_channels=128, bottleneck_ratio=0.25):
    """创建HAE V2模型"""
    model = HAEUNetModelV2(
        image_size=image_size,
        in_channels=3,
        model_channels=model_channels,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        use_hae=True,
        num_heads=8,
        num_head_channels=64,
        use_checkpoint=True,
        use_fp16=True,  # 启用半精度
        clf_free=True,
        bottleneck_ratio=bottleneck_ratio,  # V2特有参数
    )
    return model

# 使用示例
model = create_hae_v2()
print(f"HAE V2参数量: {sum(p.numel() for p in model.parameters()):,}")
```

#### 训练配置

```python
# 训练超参数（HAE V2）
training_config = {
    'lr': 2e-4,  # 更高的学习率补偿容量减少
    'batch_size': 32,  # 最大的批次
    'weight_decay': 0.005,  # 减少正则化
    'ema_rate': 0.999,  # 稍快的EMA
    'log_interval': 100,
    'save_interval': 10000,
    'microbatch': 8,
}
```

## 3. 高级配置

### 3.1 瓶颈比例调优（HAE V2）

```python
# 不同瓶颈比例的性能权衡
bottleneck_configs = {
    'ultra_efficient': 0.125,  # 最小参数量
    'efficient': 0.25,         # 默认配置
    'balanced': 0.5,           # 平衡配置
    'performance': 0.75,       # 性能优先
}

def create_hae_v2_custom(bottleneck_type='efficient'):
    ratio = bottleneck_configs[bottleneck_type]
    return create_hae_v2(bottleneck_ratio=ratio)
```

### 3.2 多尺度配置

```python
# 自定义多尺度patch大小
def create_custom_mstb_config():
    """自定义MSTB配置"""
    configs = {
        'fine_grained': [1, 2, 4],      # 细粒度
        'standard': [1, 4, 8],          # 标准配置
        'coarse_grained': [1, 8, 16],   # 粗粒度
    }
    return configs

# 注意：patch_sizes需要在模型内部修改，这里仅作配置参考
```

### 3.3 注意力配置优化

```python
def optimize_attention_config(image_size):
    """根据图像尺寸优化注意力配置"""
    if image_size <= 128:
        return {
            'attention_resolutions': [8, 4],
            'num_heads': 4,
            'num_head_channels': 32,
        }
    elif image_size <= 256:
        return {
            'attention_resolutions': [16, 8],
            'num_heads': 8,
            'num_head_channels': 64,
        }
    else:  # 512+
        return {
            'attention_resolutions': [32, 16, 8],
            'num_heads': 16,
            'num_head_channels': 64,
        }
```

## 4. 训练脚本示例

### 4.1 基础训练脚本

```python
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

def train_hae_model(model, dataloader, num_epochs=100, device='cuda'):
    """HAE模型训练函数"""
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            
            # 随机时间步
            t = torch.randint(0, 1000, (data.shape[0],), device=device)
            
            # 添加噪声
            noise = torch.randn_like(data)
            noisy_data = data + noise * 0.1  # 简化的噪声添加
            
            # 前向传播
            optimizer.zero_grad()
            pred = model(noisy_data, t)
            
            # 计算损失
            loss = F.mse_loss(pred, data)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.6f}')
```

### 4.2 分布式训练

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed_training():
    """设置分布式训练"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def train_hae_distributed(model_type='lite'):
    """分布式训练HAE模型"""
    local_rank = setup_distributed_training()
    
    # 创建模型
    if model_type == 'original':
        model = create_hae_original()
    elif model_type == 'lite':
        model = create_hae_lite()
    else:  # v2
        model = create_hae_v2()
    
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # 继续训练逻辑...
```

## 5. 推理和部署

### 5.1 模型推理

```python
def inference_hae_model(model, input_tensor, timesteps, device='cuda'):
    """HAE模型推理"""
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        timesteps = timesteps.to(device)
        
        # 推理
        output = model(input_tensor, timesteps)
        
    return output

# 使用示例
model = create_hae_lite()
model.load_state_dict(torch.load('hae_lite_checkpoint.pth'))

input_data = torch.randn(1, 3, 256, 256)
timesteps = torch.tensor([100])

result = inference_hae_model(model, input_data, timesteps)
```

### 5.2 模型优化和部署

```python
import torch.jit

def optimize_for_deployment(model, example_input, example_timesteps):
    """优化模型用于部署"""
    model.eval()
    
    # TorchScript优化
    traced_model = torch.jit.trace(
        model, 
        (example_input, example_timesteps)
    )
    
    # 保存优化后的模型
    traced_model.save('hae_optimized.pt')
    
    return traced_model

# ONNX导出
def export_to_onnx(model, example_input, example_timesteps, output_path):
    """导出为ONNX格式"""
    model.eval()
    torch.onnx.export(
        model,
        (example_input, example_timesteps),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input', 'timesteps'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'timesteps': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
```

## 6. 性能监控和调优

### 6.1 性能分析

```python
import time
import torch.profiler

def profile_hae_model(model, input_tensor, timesteps):
    """性能分析"""
    model.eval()
    
    # 预热
    for _ in range(10):
        _ = model(input_tensor, timesteps)
    
    # 计时测试
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        _ = model(input_tensor, timesteps)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"平均推理时间: {avg_time:.4f}秒")
    
    # 详细性能分析
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        _ = model(input_tensor, timesteps)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 6.2 内存使用监控

```python
def monitor_memory_usage(model, input_tensor, timesteps):
    """监控内存使用"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 前向传播
    output = model(input_tensor, timesteps)
    
    # 内存统计
    current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    print(f"当前内存使用: {current_memory:.2f} GB")
    print(f"峰值内存使用: {peak_memory:.2f} GB")
    
    return current_memory, peak_memory
```

## 7. 常见问题和解决方案

### 7.1 内存不足

```python
# 解决方案1: 启用梯度检查点
model = create_hae_lite()
model.use_checkpoint = True

# 解决方案2: 减少批次大小
batch_size = 8  # 从16减少到8

# 解决方案3: 使用混合精度
model.use_fp16 = True

# 解决方案4: 切换到HAE V2
model = create_hae_v2(bottleneck_ratio=0.125)
```

### 7.2 训练不稳定

```python
# 解决方案1: 调整学习率
optimizer = AdamW(model.parameters(), lr=5e-5)  # 降低学习率

# 解决方案2: 增加权重衰减
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.02)

# 解决方案3: 使用梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 7.3 推理速度慢

```python
# 解决方案1: 使用TorchScript
traced_model = torch.jit.trace(model, (input_tensor, timesteps))

# 解决方案2: 启用CUDA图
torch.backends.cudnn.benchmark = True

# 解决方案3: 切换到HAE V2
model = create_hae_v2()
```

## 8. 最佳实践

### 8.1 模型选择指南

```python
def choose_hae_model(gpu_memory_gb, quality_requirement, speed_requirement):
    """根据需求选择HAE模型"""
    if gpu_memory_gb >= 16 and quality_requirement == 'highest':
        return 'hae_original'
    elif gpu_memory_gb >= 8 and speed_requirement != 'critical':
        return 'hae_lite'
    else:
        return 'hae_v2'

# 使用示例
recommended_model = choose_hae_model(
    gpu_memory_gb=12, 
    quality_requirement='high', 
    speed_requirement='moderate'
)
print(f"推荐模型: {recommended_model}")
```

### 8.2 超参数调优建议

```python
# 不同模型的推荐超参数
hyperparams = {
    'hae_original': {
        'lr': 1e-4,
        'batch_size': 16,
        'weight_decay': 0.01,
        'warmup_steps': 1000,
    },
    'hae_lite': {
        'lr': 1.5e-4,
        'batch_size': 24,
        'weight_decay': 0.01,
        'warmup_steps': 800,
    },
    'hae_v2': {
        'lr': 2e-4,
        'batch_size': 32,
        'weight_decay': 0.005,
        'warmup_steps': 500,
    }
}
```

### 8.3 部署检查清单

```python
def deployment_checklist(model, test_input, test_timesteps):
    """部署前检查清单"""
    checks = {
        'model_loads': False,
        'inference_works': False,
        'output_shape_correct': False,
        'memory_usage_acceptable': False,
        'inference_time_acceptable': False,
    }
    
    try:
        # 检查模型加载
        model.eval()
        checks['model_loads'] = True
        
        # 检查推理
        with torch.no_grad():
            output = model(test_input, test_timesteps)
        checks['inference_works'] = True
        
        # 检查输出形状
        if output.shape == test_input.shape:
            checks['output_shape_correct'] = True
        
        # 检查内存使用
        memory_usage = torch.cuda.memory_allocated() / 1024**3
        if memory_usage < 8.0:  # 小于8GB
            checks['memory_usage_acceptable'] = True
        
        # 检查推理时间
        start_time = time.time()
        _ = model(test_input, test_timesteps)
        inference_time = time.time() - start_time
        if inference_time < 1.0:  # 小于1秒
            checks['inference_time_acceptable'] = True
            
    except Exception as e:
        print(f"检查过程中出现错误: {e}")
    
    # 打印检查结果
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    return all(checks.values())
```

这个使用指南提供了HAE家族模型的完整使用方法，从基础配置到高级优化，从训练到部署，涵盖了实际应用中可能遇到的各种场景和问题。