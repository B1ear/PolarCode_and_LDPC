"""
Visualization Module

绘制BER曲线和性能对比图表
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Optional


def plot_ber_curves(snr_db: np.ndarray, ber_data: Dict[str, List[float]], 
                    title: str = "BER vs SNR", save_path: Optional[str] = None,
                    show_plot: bool = True):
    """
    绘制BER曲线
    
    Args:
        snr_db: SNR值数组
        ber_data: 字典，键为标签，值为BER列表
        title: 图表标题
        save_path: 保存路径（可选）
        show_plot: 是否显示图表
    """
    plt.figure(figsize=(10, 6))
    
    for label, ber in ber_data.items():
        plt.semilogy(snr_db, ber, marker='o', label=label, linewidth=2, markersize=6)
    
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_comparison(data: Dict[str, float], ylabel: str = "Value",
                   title: str = "Performance Comparison",
                   save_path: Optional[str] = None,
                   show_plot: bool = True):
    """
    绘制柱状对比图
    
    Args:
        data: 字典，键为标签，值为数值
        ylabel: Y轴标签
        title: 图表标题
        save_path: 保存路径
        show_plot: 是否显示
    """
    plt.figure(figsize=(8, 6))
    
    labels = list(data.keys())
    values = list(data.values())
    
    plt.bar(labels, values, alpha=0.7, color=['blue', 'orange', 'green', 'red'][:len(labels)])
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def save_results(results: Dict, filepath: str):
    """
    保存结果到JSON文件
    
    Args:
        results: 结果字典
        filepath: 文件路径
    """
    # 转换numpy数组为列表
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj
    
    results_converted = convert_numpy(results)
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results_converted, f, indent=2)
    
    print(f"Results saved to {filepath}")


if __name__ == "__main__":
    # 测试代码
    print("Testing Visualization Module...")
    
    # 测试1: BER曲线
    print("\n1. BER Curves Test:")
    snr_db = np.arange(-2, 6, 1)
    ber_polar = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    ber_ldpc = [0.08, 0.04, 0.015, 0.008, 0.004, 0.0015, 0.0008, 0.0004]
    
    ber_data = {
        'Polar Code': ber_polar,
        'LDPC': ber_ldpc
    }
    
    print("Creating BER curve plot...")
    plot_ber_curves(snr_db, ber_data, show_plot=False, 
                   save_path="test_ber.png")
    
    # 测试2: 柱状图
    print("\n2. Bar Chart Test:")
    throughput_data = {
        'Polar Encoding': 100.5,
        'Polar Decoding': 20.3,
        'LDPC Encoding': 150.2,
        'LDPC Decoding': 45.8
    }
    
    print("Creating throughput comparison plot...")
    plot_comparison(throughput_data, ylabel="Throughput (Mbps)",
                   title="Encoding/Decoding Throughput",
                   show_plot=False, save_path="test_throughput.png")
    
    # 测试3: 保存结果
    print("\n3. Save Results Test:")
    results = {
        'snr_db': snr_db,
        'polar': {
            'ber': ber_polar,
            'fer': [0.15, 0.08, 0.04, 0.02, 0.01, 0.005, 0.002, 0.001]
        },
        'ldpc': {
            'ber': ber_ldpc,
            'fer': [0.12, 0.06, 0.03, 0.015, 0.008, 0.004, 0.0015, 0.0008]
        }
    }
    
    save_results(results, "test_results.json")
    
    print("\n✓ Visualization test passed!")
    print("Note: Plots saved as test_*.png files")
