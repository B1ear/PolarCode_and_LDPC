"""
吞吐量测试

测量Polar码和LDPC码的编码和解码吞吐量（比特/秒）。
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, Tuple
import time
import matplotlib.pyplot as plt

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polar import PolarEncoder, SCDecoder
from ldpc import LDPCEncoder, BPDecoder
from channel import AWGNChannel
from utils.visualization import save_results


def run_throughput_test(
    polar_config: Dict,
    ldpc_config: Dict,
    output_dir: Path,
    num_iterations: int = 100,
    snr_db: float = 3.0
) -> Dict:
    """
    测量编码和解码吞吐量
    
    Args:
        polar_config: Polar码配置
        ldpc_config: LDPC配置
        output_dir: 结果输出目录
        num_iterations: 用于计时的编码/解码循环次数
        snr_db: 信道SNR（影响解码复杂度）
        
    Returns:
        包含吞吐量结果的字典
    """
    print(f"\n{'='*60}")
    print("Throughput Test")
    print(f"{'='*60}")
    print(f"Iterations: {num_iterations}")
    print(f"SNR: {snr_db} dB")
    
    results = {
        'num_iterations': num_iterations,
        'snr_db': snr_db,
        'polar': {},
        'ldpc': {}
    }
    
    # Test Polar Code
    print(f"\n{'-'*60}")
    print("Testing Polar Code Throughput")
    print(f"{'-'*60}")
    
    polar_results = measure_polar_throughput(
        polar_config, num_iterations, snr_db
    )
    results['polar'] = polar_results
    
    # Test LDPC
    print(f"\n{'-'*60}")
    print("Testing LDPC Throughput")
    print(f"{'-'*60}")
    
    ldpc_results = measure_ldpc_throughput(
        ldpc_config, num_iterations, snr_db
    )
    results['ldpc'] = ldpc_results
    
    # Print summary
    print(f"\n{'='*60}")
    print("Throughput Summary")
    print(f"{'='*60}")
    
    print(f"\nPolar Code:")
    print(f"  Encoding:  {polar_results['encoding_throughput']:.2f} Mbps")
    print(f"  Decoding:  {polar_results['decoding_throughput']:.2f} Mbps")
    print(f"  End-to-End: {polar_results['end_to_end_throughput']:.2f} Mbps")
    
    print(f"\nLDPC:")
    print(f"  Encoding:  {ldpc_results['encoding_throughput']:.2f} Mbps")
    print(f"  Decoding:  {ldpc_results['decoding_throughput']:.2f} Mbps")
    print(f"  End-to-End: {ldpc_results['end_to_end_throughput']:.2f} Mbps")
    
    # Save results
    save_results(results, output_dir / "data" / "throughput_results.json")
    
    # Plot throughput comparison
    plot_throughput_comparison(results, output_dir)
    
    return results


def plot_throughput_comparison(results: Dict, output_dir: Path):
    """
    绘制吞吐量对比柱状图
    
    Args:
        results: 包含吞吐量测试结果的字典
        output_dir: 输出目录
    """
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 提取数据
    polar_data = results['polar']
    ldpc_data = results['ldpc']
    
    # 准备绘图数据
    categories = ['Encoding', 'Decoding', 'End-to-End']
    polar_throughput = [
        polar_data['encoding_throughput'],
        polar_data['decoding_throughput'],
        polar_data['end_to_end_throughput']
    ]
    ldpc_throughput = [
        ldpc_data['encoding_throughput'],
        ldpc_data['decoding_throughput'],
        ldpc_data['end_to_end_throughput']
    ]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(categories))
    width = 0.35
    
    # 绘制柱状图
    bars1 = ax.bar(x - width/2, polar_throughput, width, 
                   label=f'Polar (N={polar_data["N"]}, K={polar_data["K"]})', 
                   alpha=0.8, color='#3498db')
    bars2 = ax.bar(x + width/2, ldpc_throughput, width, 
                   label=f'LDPC (n={ldpc_data["n"]}, k={ldpc_data["k"]})', 
                   alpha=0.8, color='#e74c3c')
    
    # 设置标签和标题
    ax.set_xlabel('Operation Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Throughput (Mbps)', fontsize=13, fontweight='bold')
    ax.set_title(f'Polar Code vs LDPC Throughput Comparison\n(SNR={results["snr_db"]}dB, Iterations={results["num_iterations"]})', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在柱状图上添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height >= 0.001:
                label = f'{height:.4f}'
            else:
                label = f'{height:.2e}'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    label,
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_path = output_dir / "figures" / "throughput_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved throughput comparison plot: {output_path}")
    plt.close()
    
    # 打印性能对比
    print(f"\n  Performance Comparison:")
    print(f"    Polar encoding is {polar_data['encoding_throughput']/ldpc_data['encoding_throughput']:.1f}x faster than LDPC")
    print(f"    Polar decoding is {polar_data['decoding_throughput']/ldpc_data['decoding_throughput']:.1f}x faster than LDPC")
    print(f"    Polar end-to-end is {polar_data['end_to_end_throughput']/ldpc_data['end_to_end_throughput']:.1f}x faster than LDPC")


def measure_polar_throughput(
    config: Dict,
    num_iterations: int,
    snr_db: float
) -> Dict:
    """Measure Polar code throughput"""
    
    N = config['encoding']['N']
    K = config['encoding']['K']
    
    print(f"Polar: N={N}, K={K}, rate={K/N:.3f}")
    
    # Create encoder and decoder
    encoder = PolarEncoder(N, K)
    decoder = SCDecoder(N, K, frozen_bits=encoder.get_frozen_bits_positions())
    channel = AWGNChannel(snr_db=snr_db, seed=42)
    
    # Warm-up
    for _ in range(10):
        msg = np.random.randint(0, 2, K)
        cw = encoder.encode(msg)
        llr = channel.transmit(cw, return_llr=True)
        _ = decoder.decode(llr)
    
    # Measure encoding throughput
    messages = [np.random.randint(0, 2, K) for _ in range(num_iterations)]
    
    start_time = time.time()
    for msg in messages:
        _ = encoder.encode(msg)
    encoding_time = time.time() - start_time
    
    total_bits = num_iterations * K
    encoding_throughput = total_bits / encoding_time / 1e6  # Mbps
    
    print(f"  Encoding: {encoding_time:.3f}s for {num_iterations} frames")
    print(f"  Encoding Throughput: {encoding_throughput:.2f} Mbps")
    
    # Measure decoding throughput
    codewords_llr = []
    for msg in messages[:num_iterations]:
        cw = encoder.encode(msg)
        llr = channel.transmit(cw, return_llr=True)
        codewords_llr.append(llr)
    
    start_time = time.time()
    print(f"  Decoding {num_iterations} frames...", end='', flush=True)
    for llr in codewords_llr:
        _ = decoder.decode(llr)
    print(" Done!")
    decoding_time = time.time() - start_time
    
    decoding_throughput = total_bits / decoding_time / 1e6  # Mbps
    
    print(f"  Decoding: {decoding_time:.3f}s for {num_iterations} frames")
    print(f"  Decoding Throughput: {decoding_throughput:.2f} Mbps")
    
    # Measure end-to-end throughput
    start_time = time.time()
    for msg in messages[:num_iterations]:
        cw = encoder.encode(msg)
        llr = channel.transmit(cw, return_llr=True)
        _ = decoder.decode(llr)
    end_to_end_time = time.time() - start_time
    
    end_to_end_throughput = total_bits / end_to_end_time / 1e6  # Mbps
    
    print(f"  End-to-End: {end_to_end_time:.3f}s for {num_iterations} frames")
    print(f"  End-to-End Throughput: {end_to_end_throughput:.2f} Mbps")
    
    return {
        'N': N,
        'K': K,
        'rate': K / N,
        'num_iterations': num_iterations,
        'encoding_time': encoding_time,
        'decoding_time': decoding_time,
        'end_to_end_time': end_to_end_time,
        'encoding_throughput': encoding_throughput,
        'decoding_throughput': decoding_throughput,
        'end_to_end_throughput': end_to_end_throughput
    }


def measure_ldpc_throughput(
    config: Dict,
    num_iterations: int,
    snr_db: float
) -> Dict:
    """Measure LDPC throughput"""
    
    n = config['encoding']['n']
    k = config['encoding']['k']
    dv = config['encoding'].get('dv', 3)
    dc = config['encoding'].get('dc', 6)
    max_iter = config['decoding'].get('max_iterations', 50)
    
    print(f"LDPC: n={n}, k={k}, rate={k/n:.3f}, dv={dv}, dc={dc}, max_iter={max_iter}")
    
    # Create encoder and decoder
    encoder = LDPCEncoder(n, k, dv=dv, dc=dc, seed=42)
    decoder = BPDecoder(encoder.H, max_iter=max_iter)
    channel = AWGNChannel(snr_db=snr_db, seed=42)
    
    # Warm-up
    for _ in range(10):
        msg = np.random.randint(0, 2, k)
        cw = encoder.encode(msg)
        llr = channel.transmit(cw, return_llr=True)
        _ = decoder.decode(llr)
    
    # Measure encoding throughput
    messages = [np.random.randint(0, 2, k) for _ in range(num_iterations)]
    
    start_time = time.time()
    for msg in messages:
        _ = encoder.encode(msg)
    encoding_time = time.time() - start_time
    
    total_bits = num_iterations * k
    encoding_throughput = total_bits / encoding_time / 1e6  # Mbps
    
    print(f"  Encoding: {encoding_time:.3f}s for {num_iterations} frames")
    print(f"  Encoding Throughput: {encoding_throughput:.2f} Mbps")
    
    # Measure decoding throughput
    codewords_llr = []
    for msg in messages[:num_iterations]:
        cw = encoder.encode(msg)
        llr = channel.transmit(cw, return_llr=True)
        codewords_llr.append(llr)
    
    start_time = time.time()
    print(f"  Decoding {num_iterations} frames (this may take a while for LDPC)...", end='', flush=True)
    for llr in codewords_llr:
        _ = decoder.decode(llr)
    print(" Done!")
    decoding_time = time.time() - start_time
    
    decoding_throughput = total_bits / decoding_time / 1e6  # Mbps
    
    print(f"  Decoding: {decoding_time:.3f}s for {num_iterations} frames")
    print(f"  Decoding Throughput: {decoding_throughput:.2f} Mbps")
    
    # Measure end-to-end throughput
    start_time = time.time()
    for msg in messages[:num_iterations]:
        cw = encoder.encode(msg)
        llr = channel.transmit(cw, return_llr=True)
        _ = decoder.decode(llr)
    end_to_end_time = time.time() - start_time
    
    end_to_end_throughput = total_bits / end_to_end_time / 1e6  # Mbps
    
    print(f"  End-to-End: {end_to_end_time:.3f}s for {num_iterations} frames")
    print(f"  End-to-End Throughput: {end_to_end_throughput:.2f} Mbps")
    
    return {
        'n': n,
        'k': k,
        'rate': k / n,
        'num_iterations': num_iterations,
        'encoding_time': encoding_time,
        'decoding_time': decoding_time,
        'end_to_end_time': end_to_end_time,
        'encoding_throughput': encoding_throughput,
        'decoding_throughput': decoding_throughput,
        'end_to_end_throughput': end_to_end_throughput
    }


if __name__ == "__main__":
    # Test throughput measurement
    print("Testing Throughput Module...")
    
    polar_config = {
        'encoding': {'N': 128, 'K': 64},
        'construction': {'design_snr_db': 2.0}
    }
    
    ldpc_config = {
        'encoding': {'n': 120, 'k': 60, 'dv': 3, 'dc': 6},
        'decoding': {'max_iterations': 50}
    }
    
    # Use absolute path relative to this file
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    
    results = run_throughput_test(
        polar_config=polar_config,
        ldpc_config=ldpc_config,
        output_dir=output_dir,
        num_iterations=200,
        snr_db=3.0
    )
    
    print("\n✓ Throughput test passed!")
