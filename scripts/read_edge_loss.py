#!/usr/bin/env python3
"""
读取和分析 edge_loss .npy 文件的脚本

该脚本用于加载和分析由 translation_FPDM_edge_regularization.py 生成的边缘损失数据。
可以读取单个或多个 edge_loss_k.npy 文件，并提供基本的统计分析。

使用方法:
    python read_edge_loss.py --file_path /path/to/edge_loss_1.npy
    python read_edge_loss.py --dir_path /path/to/output/directory
    python read_edge_loss.py --dir_path /path/to/output/directory --plot
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_single_file(file_path):
    """
    加载单个 .npy 文件
    
    Args:
        file_path (str): .npy 文件的路径
        
    Returns:
        numpy.ndarray: 加载的数据
    """
    try:
        data = np.load(file_path)
        print(f"成功加载文件: {file_path}")
        print(f"数据形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"数据内容: {data}")
        return data
    except FileNotFoundError:
        print(f"错误: 文件未找到 - {file_path}")
        return None
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        return None


def load_directory_files(dir_path):
    """
    加载目录中所有的 edge_loss_*.npy 文件
    
    Args:
        dir_path (str): 包含 .npy 文件的目录路径
        
    Returns:
        dict: 文件名到数据的映射
    """
    if not os.path.exists(dir_path):
        print(f"错误: 目录不存在 - {dir_path}")
        return {}
    
    # 查找所有 edge_loss_*.npy 文件
    pattern = os.path.join(dir_path, "edge_loss_*.npy")
    files = glob.glob(pattern)
    
    if not files:
        print(f"在目录 {dir_path} 中未找到 edge_loss_*.npy 文件")
        return {}
    
    print(f"找到 {len(files)} 个 edge_loss 文件")
    
    data_dict = {}
    for file_path in sorted(files):
        filename = os.path.basename(file_path)
        try:
            data = np.load(file_path)
            data_dict[filename] = data
            print(f"✓ 加载 {filename}: 形状 {data.shape}, 内容 {data}")
        except Exception as e:
            print(f"✗ 加载 {filename} 失败: {e}")
    
    return data_dict


def analyze_data(data_dict):
    """
    分析加载的边缘损失数据
    
    Args:
        data_dict (dict): 文件名到数据的映射
    """
    if not data_dict:
        print("没有数据可供分析")
        return
    
    print("\n=== 数据分析 ===")
    
    all_values = []
    for filename, data in data_dict.items():
        if data.size > 0:
            all_values.extend(data.flatten())
    
    if not all_values:
        print("所有文件都是空的")
        return
    
    all_values = np.array(all_values)
    
    print(f"总数据点数: {len(all_values)}")
    print(f"平均值: {np.mean(all_values):.6f}")
    print(f"标准差: {np.std(all_values):.6f}")
    print(f"最小值: {np.min(all_values):.6f}")
    print(f"最大值: {np.max(all_values):.6f}")
    print(f"中位数: {np.median(all_values):.6f}")
    
    # 按文件显示统计信息
    print("\n=== 按文件统计 ===")
    for filename, data in data_dict.items():
        if data.size > 0:
            flat_data = data.flatten()
            print(f"{filename}:")
            print(f"  平均值: {np.mean(flat_data):.6f}")
            print(f"  标准差: {np.std(flat_data):.6f}")
            print(f"  范围: [{np.min(flat_data):.6f}, {np.max(flat_data):.6f}]")


def plot_data(data_dict, save_path=None):
    """
    绘制边缘损失数据的可视化图表
    
    Args:
        data_dict (dict): 文件名到数据的映射
        save_path (str, optional): 保存图表的路径
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("警告: matplotlib 未安装，无法绘制图表")
        return
    
    if not data_dict:
        print("没有数据可供绘制")
        return
    
    # 收集所有数据
    all_values = []
    batch_labels = []
    
    for filename, data in sorted(data_dict.items()):
        if data.size > 0:
            flat_data = data.flatten()
            all_values.extend(flat_data)
            # 从文件名提取批次号
            batch_num = filename.replace('edge_loss_', '').replace('.npy', '')
            batch_labels.extend([f'Batch {batch_num}'] * len(flat_data))
    
    if not all_values:
        print("所有文件都是空的，无法绘制图表")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 直方图
    ax1.hist(all_values, bins=20, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Edge Loss Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Edge Loss Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 时间序列图（按批次）
    batch_means = []
    batch_numbers = []
    for filename, data in sorted(data_dict.items()):
        if data.size > 0:
            batch_num = int(filename.replace('edge_loss_', '').replace('.npy', ''))
            batch_numbers.append(batch_num)
            batch_means.append(np.mean(data.flatten()))
    
    if batch_numbers:
        ax2.plot(batch_numbers, batch_means, 'o-', linewidth=2, markersize=6)
        ax2.set_xlabel('Batch Number')
        ax2.set_ylabel('Average Edge Loss')
        ax2.set_title('Edge Loss Trend Across Batches')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='读取和分析 edge_loss .npy 文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python read_edge_loss.py --file_path ./logs/edge_loss_1.npy
  python read_edge_loss.py --dir_path ./logs/output_directory
  python read_edge_loss.py --dir_path ./logs/output_directory --plot --save_plot edge_loss_analysis.png
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--file_path', 
        type=str, 
        help='单个 .npy 文件的路径'
    )
    group.add_argument(
        '--dir_path', 
        type=str, 
        help='包含 edge_loss_*.npy 文件的目录路径'
    )
    
    parser.add_argument(
        '--plot', 
        action='store_true', 
        help='绘制数据可视化图表'
    )
    parser.add_argument(
        '--save_plot', 
        type=str, 
        help='保存图表的文件路径（需要同时使用 --plot）'
    )
    
    args = parser.parse_args()
    
    if args.file_path:
        # 加载单个文件
        data = load_single_file(args.file_path)
        if data is not None:
            # 将单个文件数据转换为字典格式以便统一处理
            filename = os.path.basename(args.file_path)
            data_dict = {filename: data}
            analyze_data(data_dict)
            
            if args.plot:
                plot_data(data_dict, args.save_plot)
    
    elif args.dir_path:
        # 加载目录中的所有文件
        data_dict = load_directory_files(args.dir_path)
        if data_dict:
            analyze_data(data_dict)
            
            if args.plot:
                plot_data(data_dict, args.save_plot)
    
    if args.save_plot and not args.plot:
        print("警告: --save_plot 需要与 --plot 一起使用")


if __name__ == "__main__":
    main()