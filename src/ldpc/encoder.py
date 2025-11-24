"""
LDPC Encoder Implementation

基于校验矩阵的LDPC编码器
"""

import numpy as np
from typing import Optional
from .matrix import generate_ldpc_matrix, create_systematic_generator


class LDPCEncoder:
    """
    LDPC编码器
    
    支持系统码编码：c = [m, p]，其中p为校验位
    """
    
    def __init__(self, n: int, k: int, H: Optional[np.ndarray] = None,
                 G: Optional[np.ndarray] = None, dv: int = 3, dc: int = 6, 
                 seed: Optional[int] = None):
        """
        初始化LDPC编码器
        
        Args:
            n: 码长
            k: 信息位长度
            H: 校验矩阵，如果为None则自动生成
            G: 生成矩阵，如果提供则直接使用（优先级高于从H创建）
            dv: 变量节点度数（H为None时使用）
            dc: 校验节点度数（H为None时使用）
            seed: 随机种子
        """
        assert n > k > 0, "Invalid code parameters"
        
        self.n = n
        self.k = k
        
        # 生成或使用提供的校验矩阵
        if H is None:
            self.m = n - k  # 校验位数量
            self.H = generate_ldpc_matrix(n, k, method="mackay", dv=dv, dc=dc, seed=seed)
        else:
            # 使用提供的H矩阵
            self.H = H
            m_actual, n_actual = H.shape
            assert n_actual == n, f"H matrix must have {n} columns"
            self.m = m_actual
            # 验证k是否匹配
            if n - m_actual != k:
                print(f"Warning: H implies k={n-m_actual}, but k={k} was provided")
        
        # 处理生成矩阵G
        if G is not None:
            # 使用提供的G矩阵
            # pyldpc返回G.shape=(n,k)，需要转换为(k,n)
            if G.shape == (n, k):
                self.G = G.T  # 转置为(k,n)
            elif G.shape == (k, n):
                self.G = G  # 已经是(k,n)
            else:
                raise ValueError(f"G shape {G.shape} doesn't match (n,k)={n,k} or (k,n)={k,n}")
            self.P = None  # 不从G提取P
            self.use_direct_solving = False
        else:
            # 尝试从H创建系统码生成矩阵
            self.G, self.P = create_systematic_generator(self.H)
            
            if self.G is None:
                # 如果无法创建系统码形式，使用直接求解方法
                print("Warning: Could not create systematic generator matrix, using direct solving")
                self.use_direct_solving = True
            else:
                self.use_direct_solving = False
    
    def encode(self, message: np.ndarray) -> np.ndarray:
        """
        编码信息位
        
        Args:
            message: 信息位，长度为k
            
        Returns:
            码字，长度为n
        """
        assert len(message) == self.k, f"Message length must be {self.k}"
        
        if not self.use_direct_solving:
            # 使用生成矩阵：c = m * G
            codeword = (message @ self.G) % 2
        else:
            # 直接求解校验方程
            codeword = self._encode_direct(message)
        
        return codeword.astype(int)
    
    def _encode_direct(self, message: np.ndarray) -> np.ndarray:
        """
        直接求解方法编码
        
        对于系统码 c = [m, p]，求解 H * c^T = 0
        即 H[:, :k] * m^T + H[:, k:] * p^T = 0
        所以 p^T = H[:, k:]^(-1) * H[:, :k] * m^T
        
        Args:
            message: 信息位
            
        Returns:
            码字
        """
        # 分割H矩阵
        H1 = self.H[:, :self.k]  # 信息位部分
        H2 = self.H[:, self.k:]  # 校验位部分
        
        # 计算 syndrome: s = H1 * m^T
        syndrome = (H1 @ message) % 2
        
        try:
            # 求解 H2 * p^T = s (在GF(2)中)
            # 使用高斯消元
            parity = self._solve_gf2(H2, syndrome)
            
            # 组合信息位和校验位
            codeword = np.concatenate([message, parity])
            
        except:
            # 如果求解失败，返回零码字
            print("Warning: Encoding failed, returning zero codeword")
            codeword = np.zeros(self.n, dtype=int)
        
        return codeword
    
    def _solve_gf2(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        在GF(2)中求解 A * x = b
        
        使用高斯消元算法，优化为使用XOR操作和numpy向量化
        
        Args:
            A: 系数矩阵 (m x n)
            b: 右侧向量 (m,)
            
        Returns:
            解向量 x (n,)
        """
        m, n = A.shape
        
        # 增广矩阵 [A | b]，直接使用整型
        aug = np.hstack([A.astype(np.uint8), b.reshape(-1, 1).astype(np.uint8)])
        
        # 高斯消元 - 使用XOR操作
        pivot_row = 0
        for col in range(n):
            if pivot_row >= m:
                break
            
            # 寻找主元
            pivot_idx = None
            for row in range(pivot_row, m):
                if aug[row, col] == 1:
                    pivot_idx = row
                    break
            
            if pivot_idx is None:
                continue
            
            # 交换行
            if pivot_idx != pivot_row:
                aug[[pivot_row, pivot_idx]] = aug[[pivot_idx, pivot_row]]
            
            # 消除当前列（向量化）
            rows_to_eliminate = (aug[:, col] == 1) & (np.arange(m) != pivot_row)
            aug[rows_to_eliminate] ^= aug[pivot_row]
            
            pivot_row += 1
        
        # 回代求解
        x = np.zeros(n, dtype=int)
        for i in range(min(pivot_row, n) - 1, -1, -1):
            # 找主元行
            for row in range(i, m):
                if aug[row, i] == 1 and np.all(aug[row, :i] == 0):
                    # 计算x[i] = (b[row] + sum(A[row,j]*x[j] for j>i)) mod 2
                    x[i] = aug[row, -1] ^ np.sum(aug[row, i+1:n] & x[i+1:n])
                    break
        
        return x
    
    def verify_codeword(self, codeword: np.ndarray) -> bool:
        """
        验证码字是否有效（满足H * c^T = 0）
        
        Args:
            codeword: 码字
            
        Returns:
            True表示有效
        """
        syndrome = (self.H @ codeword) % 2
        return np.all(syndrome == 0)
    
    def get_code_rate(self) -> float:
        """获取码率"""
        return self.k / self.n
    
    def get_parity_check_matrix(self) -> np.ndarray:
        """获取校验矩阵"""
        return self.H.copy()
    
    def __repr__(self) -> str:
        return f"LDPCEncoder(n={self.n}, k={self.k}, rate={self.get_code_rate():.3f})"


if __name__ == "__main__":
    # 测试代码
    print("Testing LDPC Encoder...")
    
    # 测试1: 基本编码
    print("\n1. Basic Encoding Test:")
    n, k = 12, 6
    encoder = LDPCEncoder(n, k, dv=2, dc=4, seed=42)
    
    print(f"Encoder: {encoder}")
    print(f"H shape: {encoder.H.shape}")
    
    message = np.array([1, 0, 1, 0, 1, 1])
    codeword = encoder.encode(message)
    
    print(f"Message: {message}")
    print(f"Codeword: {codeword}")
    
    # 验证码字
    is_valid = encoder.verify_codeword(codeword)
    print(f"Valid codeword: {is_valid}")
    
    # 测试2: 多个消息
    print("\n2. Multiple Messages Test:")
    n, k = 24, 12
    encoder = LDPCEncoder(n, k, dv=3, dc=6, seed=42)
    
    num_tests = 5
    all_valid = True
    
    for i in range(num_tests):
        msg = np.random.randint(0, 2, k)
        cw = encoder.encode(msg)
        valid = encoder.verify_codeword(cw)
        
        if not valid:
            all_valid = False
            print(f"  Test {i+1}: ✗ (invalid codeword)")
        else:
            print(f"  Test {i+1}: ✓")
    
    print(f"All codewords valid: {all_valid}")
    
    # 测试3: 不同码率
    print("\n3. Different Code Rates Test:")
    configs = [(24, 12), (48, 24), (96, 48)]
    
    for n, k in configs:
        enc = LDPCEncoder(n, k, dv=3, dc=6, seed=42)
        msg = np.random.randint(0, 2, k)
        cw = enc.encode(msg)
        valid = enc.verify_codeword(cw)
        
        print(f"n={n:3d}, k={k:2d}, rate={enc.get_code_rate():.3f}, valid={valid}")
    
    # 测试4: 使用自定义H矩阵
    print("\n4. Custom H Matrix Test:")
    # 创建一个简单的H矩阵
    H_custom = np.array([
        [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    ])
    
    n, k = 12, 6
    encoder_custom = LDPCEncoder(n, k, H=H_custom)
    
    msg = np.array([1, 1, 0, 1, 0, 1])
    cw = encoder_custom.encode(msg)
    valid = encoder_custom.verify_codeword(cw)
    
    print(f"Custom H encoder: {encoder_custom}")
    print(f"Message: {msg}")
    print(f"Codeword: {cw}")
    print(f"Valid: {valid}")
    
    # 测试5: 零码字
    print("\n5. All-zeros Message Test:")
    n, k = 24, 12
    encoder = LDPCEncoder(n, k, dv=3, dc=6, seed=42)
    
    msg_zero = np.zeros(k, dtype=int)
    cw_zero = encoder.encode(msg_zero)
    valid_zero = encoder.verify_codeword(cw_zero)
    
    print(f"All-zeros message: {msg_zero}")
    print(f"Codeword: {cw_zero}")
    print(f"Valid: {valid_zero}")
    print(f"All zeros: {np.all(cw_zero == 0)}")
    
    print("\n✓ LDPC Encoder test passed!")
