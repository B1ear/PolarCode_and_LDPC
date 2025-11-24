"""
Binary Symmetric Channel (BSC) Implementation

二进制对称信道
"""

import numpy as np


class BSCChannel:
    """
    二进制对称信道
    
    以一定概率翻转比特
    """
    
    def __init__(self, crossover_prob: float, seed: int = None):
        """
        初始化BSC信道
        
        Args:
            crossover_prob: 交叉概率（翻转概率）
            seed: 随机种子
        """
        assert 0 <= crossover_prob <= 1, "Crossover probability must be in [0, 1]"
        
        self.crossover_prob = crossover_prob
        
        if seed is not None:
            np.random.seed(seed)
    
    def transmit(self, bits: np.ndarray) -> np.ndarray:
        """
        通过BSC信道传输
        
        Args:
            bits: 输入比特
            
        Returns:
            输出比特（可能有翻转）
        """
        # 生成翻转掩码
        flip_mask = np.random.random(len(bits)) < self.crossover_prob
        
        # 翻转比特
        output = bits.copy()
        output[flip_mask] = 1 - output[flip_mask]
        
        return output.astype(int)
    
    def __repr__(self) -> str:
        return f"BSCChannel(crossover_prob={self.crossover_prob})"
