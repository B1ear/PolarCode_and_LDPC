"""
LDPC Utility Functions

包含Tanner图、校验和计算等工具函数
"""

import numpy as np
from typing import List, Tuple


def create_tanner_graph(H: np.ndarray) -> Tuple[List[List[int]], List[List[int]]]:
    """
    从校验矩阵创建Tanner图的邻接表表示
    
    Args:
        H: 校验矩阵 (m x n)
        
    Returns:
        (var_neighbors, check_neighbors):
            var_neighbors[i]: 变量节点i连接的校验节点列表
            check_neighbors[j]: 校验节点j连接的变量节点列表
    """
    m, n = H.shape
    
    var_neighbors = [[] for _ in range(n)]
    check_neighbors = [[] for _ in range(m)]
    
    for i in range(m):
        for j in range(n):
            if H[i, j] == 1:
                check_neighbors[i].append(j)
                var_neighbors[j].append(i)
    
    return var_neighbors, check_neighbors


def check_syndrome(H: np.ndarray, codeword: np.ndarray) -> bool:
    """
    检查码字是否满足校验条件 H * c^T = 0
    
    Args:
        H: 校验矩阵
        codeword: 码字
        
    Returns:
        True表示满足校验条件
    """
    syndrome = (H @ codeword) % 2
    return np.all(syndrome == 0)


def calculate_syndrome(H: np.ndarray, received: np.ndarray) -> np.ndarray:
    """
    计算接收码字的伴随式
    
    Args:
        H: 校验矩阵
        received: 接收码字
        
    Returns:
        伴随式向量
    """
    return (H @ received) % 2


def count_errors(original: np.ndarray, decoded: np.ndarray) -> int:
    """
    计算比特错误数
    
    Args:
        original: 原始比特
        decoded: 解码比特
        
    Returns:
        错误比特数
    """
    return np.sum(original != decoded)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """
    计算汉明距离
    
    Args:
        a, b: 两个比特序列
        
    Returns:
        汉明距离
    """
    return np.sum(a != b)


if __name__ == "__main__":
    # 测试代码
    print("Testing LDPC Utilities...")
    
    # 测试1: Tanner图创建
    print("\n1. Tanner Graph Test:")
    H = np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ])
    
    var_neighbors, check_neighbors = create_tanner_graph(H)
    
    print(f"H shape: {H.shape}")
    print(f"Variable neighbors:")
    for i, neighbors in enumerate(var_neighbors):
        print(f"  v{i}: {neighbors}")
    
    print(f"Check neighbors:")
    for i, neighbors in enumerate(check_neighbors):
        print(f"  c{i}: {neighbors}")
    
    # 测试2: 校验和检查
    print("\n2. Syndrome Check Test:")
    codeword_valid = np.array([1, 0, 1, 1, 1, 0])
    codeword_invalid = np.array([1, 1, 1, 1, 1, 0])
    
    is_valid = check_syndrome(H, codeword_valid)
    is_invalid = check_syndrome(H, codeword_invalid)
    
    print(f"Valid codeword {codeword_valid}: {is_valid}")
    print(f"Invalid codeword {codeword_invalid}: {is_invalid}")
    
    # 测试3: 伴随式计算
    print("\n3. Syndrome Calculation Test:")
    syndrome = calculate_syndrome(H, codeword_invalid)
    print(f"Syndrome: {syndrome}")
    print(f"All zeros: {np.all(syndrome == 0)}")
    
    # 测试4: 错误计数
    print("\n4. Error Counting Test:")
    original = np.array([1, 0, 1, 0, 1])
    decoded1 = np.array([1, 0, 1, 0, 1])
    decoded2 = np.array([1, 1, 1, 0, 0])
    
    errors1 = count_errors(original, decoded1)
    errors2 = count_errors(original, decoded2)
    
    print(f"Original: {original}")
    print(f"Decoded1: {decoded1}, errors: {errors1}")
    print(f"Decoded2: {decoded2}, errors: {errors2}")
    
    # 测试5: 汉明距离
    print("\n5. Hamming Distance Test:")
    a = np.array([1, 0, 1, 0, 1, 1, 0])
    b = np.array([1, 1, 1, 0, 0, 1, 0])
    
    dist = hamming_distance(a, b)
    print(f"a: {a}")
    print(f"b: {b}")
    print(f"Hamming distance: {dist}")
    
    print("\n✓ LDPC Utilities test passed!")
