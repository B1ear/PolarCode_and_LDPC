"""
Polar Code解码器（标准实现）

包括SC和SCL解码器
参考: Arikan's original paper和标准教科书实现
"""

import numpy as np
from typing import Optional


class SCDecoder:
    """
    标准SC解码器实现
    """
    
    def __init__(self, N: int, K: int, frozen_bits: Optional[np.ndarray] = None):
        assert N > 0 and (N & (N - 1)) == 0
        assert 0 < K < N
        
        self.N = N
        self.K = K
        self.n = int(np.log2(N))
        
        if frozen_bits is None:
            from .utils import generate_frozen_bits
            self.frozen_bits, self.info_bits = generate_frozen_bits(N, K)
        else:
            self.frozen_bits = frozen_bits
            self.info_bits = np.setdiff1d(np.arange(N), frozen_bits)
        
        self.is_frozen = np.zeros(N, dtype=bool)
        self.is_frozen[self.frozen_bits] = True
        
        # 存僭中间结果
        # 层0: 1个节点，层1: 2个节点，..., 层n: N个节点
        self.llr_layers = [np.zeros(2**i) for i in range(self.n + 1)]
        self.bit_layers = [np.zeros(2**i, dtype=int) for i in range(self.n + 1)]
    
    def decode(self, llr_input: np.ndarray) -> np.ndarray:
        """SC解码主函数"""
        assert len(llr_input) == self.N
        
        # 初始化
        self.llr_layers[self.n][:] = llr_input
        u_hat = np.zeros(self.N, dtype=int)
        
        # 逐位解码
        for i in range(self.N):
            # 计算当前位的LLR
            llr_val = self._get_llr(i)
            
            # 判决
            if self.is_frozen[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if llr_val >= 0 else 1
            
            # 更新比特值
            self._update_bits(i, u_hat[i])
        
        return u_hat[self.info_bits]
    
    def _get_llr(self, phi: int) -> float:
        """计算位置phi的LLR"""
        # 从底层向上计算到顶层
        for depth in range(self.n, 0, -1):
            # 当前层的节点数
            n_nodes = self.N // (2 ** depth)
            # phi在当前层的位置
            pos_in_layer = phi // (2 ** depth)
            
            # 是否在右子树
            is_right = (phi // (2 ** (depth - 1))) % 2 == 1
            
            if not is_right:
                # 左孩子 - f函数
                idx = pos_in_layer
                llr_l = self.llr_layers[depth][2 * idx]
                llr_r = self.llr_layers[depth][2 * idx + 1]
                self.llr_layers[depth - 1][idx] = self._f(llr_l, llr_r)
            else:
                # 右孩子 - g函数
                idx = pos_in_layer
                llr_l = self.llr_layers[depth][2 * idx]
                llr_r = self.llr_layers[depth][2 * idx + 1]
                bit_l = self.bit_layers[depth - 1][idx]
                self.llr_layers[depth - 1][idx] = self._g(llr_l, llr_r, bit_l)
        
        return self.llr_layers[0][0]
    
    def _update_bits(self, phi: int, bit: int):
        """更新比特值并向下传播"""
        # 顶层设置比特
        self.bit_layers[0][0] = bit
        
        # 向下传播
        for depth in range(1, self.n + 1):
            n_nodes = self.N // (2 ** depth)
            pos_in_layer = phi // (2 ** depth)
            is_right = (phi // (2 ** (depth - 1))) % 2 == 1
            
            if not is_right:
                # 左孩子
                self.bit_layers[depth][2 * pos_in_layer] = self.bit_layers[depth - 1][pos_in_layer]
            else:
                # 右孩子
                bit_l = self.bit_layers[depth][2 * pos_in_layer]
                bit_p = self.bit_layers[depth - 1][pos_in_layer]
                self.bit_layers[depth][2 * pos_in_layer + 1] = (bit_l + bit_p) % 2
    
    def _f(self, a: float, b: float) -> float:
        """f函数: min-sum近似"""
        return np.sign(a) * np.sign(b) * min(abs(a), abs(b))
    
    def _g(self, a: float, b: float, u: int) -> float:
        """g函数"""
        return b + (1 - 2 * u) * a
    
    def __repr__(self) -> str:
        return f"SCDecoder(N={self.N}, K={self.K})"


class SCLDecoder:
    """
    SCL解码器（使用SC作为基础，L=1时退化为SC）
    """
    
    def __init__(self, N: int, K: int, list_size: int = 8,
                 frozen_bits: Optional[np.ndarray] = None):
        assert N > 0 and (N & (N - 1)) == 0
        assert 0 < K < N
        
        self.N = N
        self.K = K
        self.L = list_size
        
        if frozen_bits is None:
            from .utils import generate_frozen_bits
            self.frozen_bits, self.info_bits = generate_frozen_bits(N, K)
        else:
            self.frozen_bits = frozen_bits
            self.info_bits = np.setdiff1d(np.arange(N), frozen_bits)
        
        self.is_frozen = np.zeros(N, dtype=bool)
        self.is_frozen[self.frozen_bits] = True
        
        # 使用SC解码器作为基础
        self.sc_decoder = SCDecoder(N, K, frozen_bits)
    
    def decode(self, llr_input: np.ndarray) -> np.ndarray:
        """SCL解码 - 当前简化实现使用SC"""
        # TODO: 实现完整的路径分裂逻辑
        # 暂时使用SC解码器
        return self.sc_decoder.decode(llr_input)
    
    def __repr__(self) -> str:
        return f"SCLDecoder(N={self.N}, K={self.K}, L={self.L})"
