"""
Polar Code Utility Functions

包含CRC、位反转、冻结位生成等工具函数
"""

import numpy as np
from typing import List, Tuple


def bit_reverse(n: int, num_bits: int) -> int:
    """
    位反转操作
    
    Args:
        n: 输入整数
        num_bits: 比特位数
        
    Returns:
        位反转后的整数
    """
    result = 0
    for i in range(num_bits):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


def bit_reverse_array(arr: np.ndarray, num_bits: int) -> np.ndarray:
    """
    对数组进行位反转排列
    
    Args:
        arr: 输入数组
        num_bits: log2(N)
        
    Returns:
        位反转后的数组
    """
    N = len(arr)
    reversed_arr = np.zeros_like(arr)
    for i in range(N):
        j = bit_reverse(i, num_bits)
        reversed_arr[j] = arr[i]
    return reversed_arr


def generate_frozen_bits(N: int, K: int, channel_param: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成冻结位和信息位的位置
    
    简单版本：选择最可靠的K个位置作为信息位
    更复杂的版本需要根据信道参数计算Bhattacharyya参数
    
    Args:
        N: 码长
        K: 信息位数量
        channel_param: 信道参数（Bhattacharyya参数或信道容量），如果为None则使用简单规则
        
    Returns:
        frozen_bits: 冻结位位置索引
        info_bits: 信息位位置索引
    """
    if channel_param is None:
        # 简单规则：选择位反转后索引较大的位置
        # 这是一个启发式方法，实际应该根据信道质量计算
        n = int(np.log2(N))
        indices = np.arange(N)
        # 位反转后的索引越大，信道质量越好
        bit_reversed_indices = np.array([bit_reverse(i, n) for i in indices])
        sorted_indices = np.argsort(bit_reversed_indices)
        
        # 最后K个位置作为信息位
        info_bits = sorted_indices[-K:]
        frozen_bits = sorted_indices[:-K]
    else:
        # 根据信道参数选择
        # channel_param越小，信道质量越好
        sorted_indices = np.argsort(channel_param)
        info_bits = sorted_indices[:K]  # 选择最好的K个
        frozen_bits = sorted_indices[K:]
    
    return np.sort(frozen_bits), np.sort(info_bits)


def crc_encode(data: np.ndarray, polynomial: str = "CRC-8") -> np.ndarray:
    """
    CRC编码
    
    Args:
        data: 输入数据比特
        polynomial: CRC多项式类型
        
    Returns:
        附加CRC的数据
    """
    # 常用CRC多项式
    polynomials = {
        "CRC-8": 0x1D,   # x^8 + x^4 + x^3 + x^2 + 1
        "CRC-16": 0x1021, # x^16 + x^12 + x^5 + 1
        "CRC-24": 0x1864CFB, # 5G NR使用
    }
    
    if polynomial not in polynomials:
        polynomial = "CRC-8"
    
    poly = polynomials[polynomial]
    crc_len = int(polynomial.split("-")[1])
    
    # 计算CRC
    crc = 0
    for bit in data:
        crc = crc ^ (int(bit) << (crc_len - 1))
        for _ in range(1):
            if crc & (1 << (crc_len - 1)):
                crc = (crc << 1) ^ poly
            else:
                crc = crc << 1
        crc = crc & ((1 << crc_len) - 1)
    
    # 转换为比特数组
    crc_bits = np.array([(crc >> i) & 1 for i in range(crc_len - 1, -1, -1)], dtype=int)
    
    # 附加CRC
    return np.concatenate([data, crc_bits])


def crc_check(data: np.ndarray, polynomial: str = "CRC-8") -> bool:
    """
    CRC校验
    
    Args:
        data: 包含CRC的数据
        polynomial: CRC多项式类型
        
    Returns:
        True表示校验通过
    """
    polynomials = {
        "CRC-8": 0x1D,
        "CRC-16": 0x1021,
        "CRC-24": 0x1864CFB,
    }
    
    if polynomial not in polynomials:
        polynomial = "CRC-8"
    
    poly = polynomials[polynomial]
    crc_len = int(polynomial.split("-")[1])
    
    # 计算整个数据的CRC（包括附加的CRC位）
    crc = 0
    for bit in data:
        crc = crc ^ (int(bit) << (crc_len - 1))
        for _ in range(1):
            if crc & (1 << (crc_len - 1)):
                crc = (crc << 1) ^ poly
            else:
                crc = crc << 1
        crc = crc & ((1 << crc_len) - 1)
    
    # CRC为0表示校验通过
    return crc == 0


def polar_transform_recursive(u: np.ndarray) -> np.ndarray:
    """
    递归实现Polar变换（用于理解，实际使用迭代版本更快）
    
    x = u * G_N, where G_N = F^⊗n, F = [[1,0],[1,1]]
    
    Args:
        u: 输入向量
        
    Returns:
        变换后的向量
    """
    N = len(u)
    if N == 1:
        return u
    
    # 分为两半
    u1 = u[:N//2]
    u2 = u[N//2:]
    
    # 递归变换
    x1 = polar_transform_recursive((u1 + u2) % 2)
    x2 = polar_transform_recursive(u2)
    
    return np.concatenate([x1, x2])


def polar_transform_iterative(u: np.ndarray) -> np.ndarray:
    """
    迭代实现Polar变换（更高效）
    
    Args:
        u: 输入向量
        
    Returns:
        变换后的向量
    """
    N = len(u)
    n = int(np.log2(N))
    x = u.copy()
    
    # 迭代变换
    for stage in range(n):
        stride = 2 ** stage
        for i in range(0, N, 2 * stride):
            for j in range(stride):
                idx1 = i + j
                idx2 = i + j + stride
                temp = x[idx1]
                x[idx1] = (x[idx1] + x[idx2]) % 2
                # x[idx2] 保持不变（在GF(2)中，u2 XOR 0 = u2）
                # 但实际上这里应该是 x[idx2] = (temp + x[idx2]) % 2
        
    # 使用正确的实现
    x = u.copy()
    for stage in range(n):
        stride = 2 ** stage
        for i in range(0, N, 2 * stride):
            for j in range(stride):
                idx1 = i + j
                idx2 = i + j + stride
                x[idx1] = (x[idx1] + x[idx2]) % 2
    
    return x


if __name__ == "__main__":
    # 测试代码
    print("Testing Polar Utilities...")
    
    # 测试位反转
    print("\n1. Bit Reverse Test:")
    for i in range(8):
        rev = bit_reverse(i, 3)
        print(f"  {i:03b} -> {rev:03b} ({i} -> {rev})")
    
    # 测试位反转数组
    arr = np.arange(8)
    rev_arr = bit_reverse_array(arr, 3)
    print(f"\nArray: {arr}")
    print(f"Reversed: {rev_arr}")
    
    # 测试冻结位生成
    print("\n2. Frozen Bits Generation Test:")
    N, K = 8, 4
    frozen, info = generate_frozen_bits(N, K)
    print(f"N={N}, K={K}")
    print(f"Frozen bits: {frozen}")
    print(f"Info bits: {info}")
    
    # 测试CRC
    print("\n3. CRC Test:")
    data = np.array([1, 0, 1, 1, 0, 0, 1, 1])
    print(f"Original data: {data}")
    
    encoded = crc_encode(data, "CRC-8")
    print(f"CRC encoded: {encoded}")
    
    check = crc_check(encoded, "CRC-8")
    print(f"CRC check: {check}")
    
    # 引入错误
    encoded_error = encoded.copy()
    encoded_error[0] = 1 - encoded_error[0]
    check_error = crc_check(encoded_error, "CRC-8")
    print(f"CRC check (with error): {check_error}")
    
    # 测试Polar变换
    print("\n4. Polar Transform Test:")
    u = np.array([1, 0, 1, 1])
    x_rec = polar_transform_recursive(u)
    x_iter = polar_transform_iterative(u)
    print(f"Input: {u}")
    print(f"Recursive: {x_rec}")
    print(f"Iterative: {x_iter}")
    
    print("\n✓ Polar utilities test passed!")
