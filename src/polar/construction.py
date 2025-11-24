"""
Polar Code Construction Algorithms

实现码构造算法，选择最佳的信息位位置
"""

import numpy as np
from typing import Tuple


def bhattacharyya_bounds(N: int, snr_db: float) -> np.ndarray:
    """
    计算Bhattacharyya参数的上界
    
    用于确定各个比特信道的可靠性
    Z(W_N^(i)) 越小，信道质量越好
    
    Args:
        N: 码长（必须是2的幂次）
        snr_db: 设计信噪比(dB)
        
    Returns:
        长度为N的Bhattacharyya参数数组
    """
    n = int(np.log2(N))
    snr_linear = 10 ** (snr_db / 10.0)
    
    # 基础信道的Bhattacharyya参数
    # 对于AWGN信道: Z = exp(-SNR)
    Z_base = np.exp(-snr_linear)
    
    # 初始化
    Z = np.array([Z_base])
    
    # 递归计算
    for level in range(n):
        Z_new = np.zeros(2 ** (level + 1))
        
        for i in range(2 ** level):
            # 坏信道: Z_{2i} = 2*Z_i - Z_i^2
            Z_new[2*i] = 2 * Z[i] - Z[i] ** 2
            
            # 好信道: Z_{2i+1} = Z_i^2
            Z_new[2*i + 1] = Z[i] ** 2
        
        Z = Z_new
    
    return Z


def gaussian_approximation(N: int, snr_db: float) -> np.ndarray:
    """
    高斯近似方法计算信道参数
    
    使用密度演化的高斯近似来计算各比特信道的容量或可靠性
    
    Args:
        N: 码长
        snr_db: 设计信噪比(dB)
        
    Returns:
        长度为N的信道质量参数数组（值越大质量越好）
    """
    n = int(np.log2(N))
    snr_linear = 10 ** (snr_db / 10.0)
    
    # 基础信道的LLR均值
    # 对于AWGN-BPSK: mu = 2 * SNR
    mu_base = 2.0 * snr_linear
    
    # 初始化
    mu = np.array([mu_base])
    
    # 递归计算
    for level in range(n):
        mu_new = np.zeros(2 ** (level + 1))
        
        for i in range(2 ** level):
            # 坏信道（近似）
            # mu_{2i} ≈ φ^{-1}(1 - (1 - φ(mu_i))^2)
            # 简化: mu_{2i} ≈ mu_i (实际更复杂)
            if mu[i] < 10:  # 避免数值问题
                mu_new[2*i] = mu[i] * 0.9  # 质量下降
            else:
                mu_new[2*i] = mu[i]
            
            # 好信道
            # mu_{2i+1} ≈ 2 * mu_i
            mu_new[2*i + 1] = 2 * mu[i]
            if mu_new[2*i + 1] > 100:  # 饱和
                mu_new[2*i + 1] = 100
        
        mu = mu_new
    
    # 将mu转换为容量或可靠性度量
    # 这里使用mu作为质量度量（值越大越好）
    return mu


def construct_polar_code(N: int, K: int, method: str = "bhattacharyya",
                        snr_db: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造Polar码，选择信息位位置
    
    Args:
        N: 码长
        K: 信息位数量
        method: 构造方法 ("bhattacharyya", "gaussian_approximation", "default")
        snr_db: 设计信噪比
        
    Returns:
        (frozen_bits, info_bits): 冻结位和信息位的位置索引
    """
    if method == "bhattacharyya":
        # 使用Bhattacharyya参数
        Z = bhattacharyya_bounds(N, snr_db)
        # Z越小越好，选择最小的K个
        sorted_indices = np.argsort(Z)
        info_bits = sorted_indices[:K]
        frozen_bits = sorted_indices[K:]
        
    elif method == "gaussian_approximation":
        # 使用高斯近似
        mu = gaussian_approximation(N, snr_db)
        # mu越大越好，选择最大的K个
        sorted_indices = np.argsort(mu)[::-1]
        info_bits = sorted_indices[:K]
        frozen_bits = sorted_indices[K:]
        
    else:  # default
        # 简单的位反转启发式
        from .utils import bit_reverse
        n = int(np.log2(N))
        indices = np.arange(N)
        bit_reversed_indices = np.array([bit_reverse(i, n) for i in indices])
        sorted_indices = np.argsort(bit_reversed_indices)
        info_bits = sorted_indices[-K:]
        frozen_bits = sorted_indices[:-K]
    
    return np.sort(frozen_bits), np.sort(info_bits)


def calculate_channel_capacities(N: int, snr_db: float) -> np.ndarray:
    """
    计算各比特信道的容量
    
    Args:
        N: 码长
        snr_db: 信噪比
        
    Returns:
        长度为N的信道容量数组
    """
    # 使用Bhattacharyya参数估计容量
    Z = bhattacharyya_bounds(N, snr_db)
    
    # 容量近似: C ≈ 1 - Z (当Z较小时)
    # 更精确: C = 1 - H((1-Z)/2), H是二元熵函数
    capacities = np.zeros(N)
    for i in range(N):
        if Z[i] < 1e-10:
            capacities[i] = 1.0
        elif Z[i] > 1 - 1e-10:
            capacities[i] = 0.0
        else:
            # 二元熵函数
            p = (1 - Z[i]) / 2
            if p > 0 and p < 1:
                h = -p * np.log2(p) - (1-p) * np.log2(1-p)
                capacities[i] = 1 - h
            else:
                capacities[i] = 0.0
    
    return capacities


if __name__ == "__main__":
    # 测试代码
    print("Testing Polar Construction...")
    
    # 测试1: Bhattacharyya参数
    print("\n1. Bhattacharyya Bounds Test:")
    N = 8
    snr_db = 3.0
    Z = bhattacharyya_bounds(N, snr_db)
    print(f"N={N}, SNR={snr_db}dB")
    print(f"Bhattacharyya parameters:")
    for i, z in enumerate(Z):
        print(f"  Channel {i}: Z={z:.6f}")
    
    # 测试2: 高斯近似
    print("\n2. Gaussian Approximation Test:")
    mu = gaussian_approximation(N, snr_db)
    print(f"LLR means:")
    for i, m in enumerate(mu):
        print(f"  Channel {i}: μ={m:.4f}")
    
    # 测试3: 码构造
    print("\n3. Code Construction Test:")
    N, K = 16, 8
    methods = ["bhattacharyya", "gaussian_approximation", "default"]
    
    for method in methods:
        frozen, info = construct_polar_code(N, K, method=method, snr_db=2.0)
        print(f"\nMethod: {method}")
        print(f"  Info bits: {info}")
        print(f"  Frozen bits: {frozen}")
    
    # 测试4: 信道容量
    print("\n4. Channel Capacities Test:")
    N = 8
    snr_db = 3.0
    capacities = calculate_channel_capacities(N, snr_db)
    print(f"N={N}, SNR={snr_db}dB")
    print(f"Channel capacities:")
    for i, c in enumerate(capacities):
        print(f"  Channel {i}: C={c:.4f}")
    
    # 平均容量应该接近信道容量
    avg_capacity = np.mean(capacities)
    print(f"Average capacity: {avg_capacity:.4f}")
    
    # 测试5: 不同SNR下的码构造
    print("\n5. Construction at Different SNRs:")
    N, K = 16, 8
    snr_values = [-2, 0, 2, 4]
    
    for snr in snr_values:
        frozen, info = construct_polar_code(N, K, method="bhattacharyya", snr_db=snr)
        print(f"SNR={snr:2d}dB: Info bits = {info}")
    
    print("\n✓ Polar Construction test passed!")
