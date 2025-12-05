"""
复杂度分析

估算Polar码和LDPC码的计算复杂度（操作数、内存使用）。
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.visualization import save_results


def analyze_complexity(
    polar_config: Dict,
    ldpc_config: Dict,
    output_dir: Path
) -> Dict:
    """
    分析Polar码和LDPC实现的计算复杂度
    
    Args:
        polar_config: Polar码配置
        ldpc_config: LDPC配置
        output_dir: 结果输出目录
        
    Returns:
        包含复杂度估算的字典
    """
    print(f"\n{'='*60}")
    print("Complexity Analysis")
    print(f"{'='*60}")
    
    results = {
        'polar': {},
        'ldpc': {}
    }
    
    # 分析Polar码复杂度
    print(f"\n{'-'*60}")
    print("Analyzing Polar Code Complexity")
    print(f"{'-'*60}")
    
    polar_complexity = analyze_polar_complexity(polar_config)
    results['polar'] = polar_complexity
    
    # 分析LDPC复杂度
    print(f"\n{'-'*60}")
    print("Analyzing LDPC Complexity")
    print(f"{'-'*60}")
    
    ldpc_complexity = analyze_ldpc_complexity(ldpc_config)
    results['ldpc'] = ldpc_complexity
    
    # 打印摘要
    print(f"\n{'='*60}")
    print("Complexity Summary")
    print(f"{'='*60}")
    
    print(f"\nPolar Code (N={polar_complexity['N']}, K={polar_complexity['K']}):")
    print(f"  Encoding Operations:  {polar_complexity['encoding_ops']:,}")
    print(f"  Decoding Operations:  {polar_complexity['decoding_ops']:,}")
    print(f"  Encoding Memory:      {polar_complexity['encoding_memory']:,} bits")
    print(f"  Decoding Memory:      {polar_complexity['decoding_memory']:,} bits")
    
    print(f"\nLDPC (n={ldpc_complexity['n']}, k={ldpc_complexity['k']}):")
    print(f"  Encoding Operations:  {ldpc_complexity['encoding_ops']:,}")
    print(f"  Decoding Operations:  {ldpc_complexity['decoding_ops']:,}")
    print(f"  Encoding Memory:      {ldpc_complexity['encoding_memory']:,} bits")
    print(f"  Decoding Memory:      {ldpc_complexity['decoding_memory']:,} bits")
    
    # 绘制对比图
    plot_complexity_comparison(results, output_dir)
    
    # 保存结果
    save_results(results, output_dir / "data" / "complexity_results.json")
    
    return results


def analyze_polar_complexity(config: Dict) -> Dict:
    """分析Polar码复杂度"""
    
    N = config['encoding']['N']
    K = config['encoding']['K']
    n = int(np.log2(N))  # log2(N)
    
    print(f"Polar: N={N}, K={K}, n=log2(N)={n}")
    
    # 编码复杂度
    # Polar编码: O(N log N) 异或操作
    # 每个阶段有N/2个蝶形操作（每个2次异或）
    # 共有log2(N)个阶段
    encoding_ops = N * n  # 总异或操作数
    
    print(f"\nEncoding Complexity:")
    print(f"  Algorithm: Successive Butterfly (iterative)")
    print(f"  Stages: {n}")
    print(f"  Operations per stage: {N} XORs")
    print(f"  Total XOR operations: {encoding_ops}")
    print(f"  Time Complexity: O(N log N)")
    
    # 编码内存
    # 需要存储: 输入向量(N), 输出向量(N)
    encoding_memory = 2 * N  # bits
    
    print(f"  Memory: {encoding_memory} bits ({encoding_memory/8:.0f} bytes)")
    
    # 解码复杂度（SC解码器 - 硬判决 + 逆变换）
    # 硬判决: N次比较
    # 逆变换: N log N次异或（与编码相同）
    decoding_ops_hard = N  # 比较次数
    decoding_ops_transform = N * n  # 逆变换的异或次数
    decoding_ops = decoding_ops_hard + decoding_ops_transform
    
    print(f"\nDecoding Complexity (SC - Hard Decision + Inverse Transform):")
    print(f"  Hard decision: {decoding_ops_hard} comparisons")
    print(f"  Inverse transform: {decoding_ops_transform} XORs")
    print(f"  Total operations: {decoding_ops}")
    print(f"  Time Complexity: O(N log N)")
    
    # 解码内存
    # 需要: 接收LLR(N), 解码比特(N)
    decoding_memory = 2 * N  # 比特
    
    print(f"  Memory: {decoding_memory} bits ({decoding_memory/8:.0f} bytes)")
    
    # SCL解码器复杂度（供参考，虽然我们不使用它）
    L = 8  # 典型列表大小
    scl_ops = L * N * n  # L倍SC复杂度
    scl_memory = L * 2 * N  # L份SC内存
    
    print(f"\nSCL Decoder Complexity (L={L}, for reference):")
    print(f"  Operations: {scl_ops:,} (L × SC complexity)")
    print(f"  Memory: {scl_memory:,} bits ({scl_memory/8:.0f} bytes)")
    
    return {
        'N': N,
        'K': K,
        'n': n,
        'rate': K / N,
        'encoding_ops': encoding_ops,
        'encoding_memory': encoding_memory,
        'encoding_complexity': f"O(N log N) = O({N} × {n})",
        'decoding_ops': decoding_ops,
        'decoding_memory': decoding_memory,
        'decoding_complexity': f"O(N log N) = O({N} × {n})",
        'scl_ops': scl_ops,
        'scl_memory': scl_memory
    }


def analyze_ldpc_complexity(config: Dict) -> Dict:
    """分析LDPC复杂度"""
    
    n = config['encoding']['n']
    k = config['encoding']['k']
    m = n - k
    dv = config['encoding'].get('dv', 3)
    dc = config['encoding'].get('dc', 6)
    max_iter = config['decoding'].get('max_iterations', 50)
    
    print(f"LDPC: n={n}, k={k}, m={m}, dv={dv}, dc={dc}")
    
    # 编码复杂度
    # 对于系统码: O(m * k) 操作
    # 需要计算校验位: p = H2^-1 * H1 * m
    encoding_ops = m * k  # 矩阵向量乘法
    
    print(f"\n编码复杂度:")
    print(f"  算法: 系统码编码 (H * c = 0)")
    print(f"  矩阵向量乘法: {m} × {k}")
    print(f"  操作数: ~{encoding_ops}")
    print(f"  时间复杂度: O(m × k)")
    
    # 编码内存
    # 需要: 消息(k), 码字(n), H矩阵(m × n)
    encoding_memory = k + n + (m * n)
    
    print(f"  Memory: ~{encoding_memory} bits ({encoding_memory/8:.0f} bytes)")
    print(f"    Message: {k} bits")
    print(f"    Codeword: {n} bits")
    print(f"    H matrix: {m}×{n} = {m*n} bits")
    
    # 解码复杂度（BP解码器）
    # 每次迭代:
    #   - 变量节点更新: O(n × dv) 操作
    #   - 校验节点更新: O(m × dc) 操作
    # 总计: O(max_iter × (n × dv + m × dc))
    
    ops_per_var_node = dv * 2  # 每条边的加法和减法
    ops_per_check_node = dc * 3  # 每条边的乘积、最小值、符号
    
    ops_per_iter = n * ops_per_var_node + m * ops_per_check_node
    decoding_ops = max_iter * ops_per_iter
    
    print(f"\nDecoding Complexity (BP Decoder):")
    print(f"  Algorithm: Belief Propagation (Sum-Product)")
    print(f"  Max iterations: {max_iter}")
    print(f"  Operations per iteration:")
    print(f"    Variable nodes: {n} × {ops_per_var_node} = {n * ops_per_var_node}")
    print(f"    Check nodes: {m} × {ops_per_check_node} = {m * ops_per_check_node}")
    print(f"    Total per iter: {ops_per_iter}")
    print(f"  Total operations: ~{decoding_ops:,}")
    print(f"  Time Complexity: O(I × (n × dv + m × dc))")
    
    # 解码内存
    # 需要: LLR(n), 消息v2c(n × dv), 消息c2v(m × dc)
    num_edges = n * dv  # = m * dc
    decoding_memory = n + 2 * num_edges  # LLR + 两个消息数组
    
    print(f"  Memory: ~{decoding_memory} bits ({decoding_memory/8:.0f} bytes)")
    print(f"    LLRs: {n} values")
    print(f"    Messages: 2 × {num_edges} = {2*num_edges} values")
    
    return {
        'n': n,
        'k': k,
        'm': m,
        'dv': dv,
        'dc': dc,
        'max_iter': max_iter,
        'rate': k / n,
        'encoding_ops': encoding_ops,
        'encoding_memory': encoding_memory,
        'encoding_complexity': f"O(m × k) = O({m} × {k})",
        'decoding_ops': decoding_ops,
        'decoding_memory': decoding_memory,
        'decoding_complexity': f"O(I × (n×dv + m×dc)) = O({max_iter} × {ops_per_iter})"
    }


def plot_complexity_comparison(results: Dict, output_dir: Path):
    """绘制复杂度对比图"""
    
    polar = results['polar']
    ldpc = results['ldpc']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 操作数对比
    ax = axes[0]
    
    categories = ['Encoding', 'Decoding']
    polar_ops = [polar['encoding_ops'], polar['decoding_ops']]
    ldpc_ops = [ldpc['encoding_ops'], ldpc['decoding_ops']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, polar_ops, width, label='Polar', alpha=0.8)
    ax.bar(x + width/2, ldpc_ops, width, label='LDPC', alpha=0.8)
    
    ax.set_xlabel('Operation', fontsize=12)
    ax.set_ylabel('Number of Operations', fontsize=12)
    ax.set_title('Computational Complexity', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(polar_ops):
        ax.text(i - width/2, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(ldpc_ops):
        ax.text(i + width/2, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
    
    # 内存对比
    ax = axes[1]
    
    polar_mem = [polar['encoding_memory'], polar['decoding_memory']]
    ldpc_mem = [ldpc['encoding_memory'], ldpc['decoding_memory']]
    
    ax.bar(x - width/2, polar_mem, width, label='Polar', alpha=0.8)
    ax.bar(x + width/2, ldpc_mem, width, label='LDPC', alpha=0.8)
    
    ax.set_xlabel('Operation', fontsize=12)
    ax.set_ylabel('Memory (bits)', fontsize=12)
    ax.set_title('Memory Usage', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(polar_mem):
        ax.text(i - width/2, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(ldpc_mem):
        ax.text(i + width/2, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    complexity_path = output_dir / "figures" / "complexity_comparison.png"
    plt.savefig(complexity_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved complexity plot: {complexity_path}")
    plt.close()


if __name__ == "__main__":
    # Test complexity analysis
    print("Testing Complexity Analysis Module...")
    
    polar_config = {
        'encoding': {'N': 128, 'K': 64},
        'construction': {'design_snr_db': 2.0}
    }
    
    ldpc_config = {
        'encoding': {'n': 120, 'k': 60, 'dv': 3, 'dc': 6},
        'decoding': {'max_iterations': 50}
    }
    
    # 使用相对于此文件的绝对路径
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    
    results = analyze_complexity(
        polar_config=polar_config,
        ldpc_config=ldpc_config,
        output_dir=output_dir
    )
    
    print("\n✓ Complexity analysis test passed!")
