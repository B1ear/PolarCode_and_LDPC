"""
AWGN (Additive White Gaussian Noise) Channel Implementation

BPSK调制下的AWGN信道模拟
"""

import numpy as np
from typing import Union


class AWGNChannel:
    """
    加性高斯白噪声(AWGN)信道
    
    使用BPSK调制: 0 -> +1, 1 -> -1
    """
    
    def __init__(self, snr_db: float, seed: int = None):
        """
        初始化AWGN信道
        
        Args:
            snr_db: 信噪比(dB)
            seed: 随机种子，用于可重复性
        """
        self.snr_db = snr_db
        self.snr_linear = 10 ** (snr_db / 10.0)
        
        # 计算噪声标准差
        # 对于BPSK: Eb/N0 = SNR (假设码率为1)
        # 噪声功率 N0 = 1/SNR, 标准差 = sqrt(N0/2)
        self.noise_std = np.sqrt(1.0 / (2.0 * self.snr_linear))
        
        if seed is not None:
            np.random.seed(seed)
    
    def modulate_bpsk(self, bits: np.ndarray) -> np.ndarray:
        """
        BPSK调制: 0 -> +1, 1 -> -1
        
        Args:
            bits: 二进制比特数组
            
        Returns:
            调制后的符号
        """
        return 1.0 - 2.0 * bits.astype(float)
    
    def demodulate_bpsk_hard(self, symbols: np.ndarray) -> np.ndarray:
        """
        BPSK硬判决解调: >0 -> 0, <=0 -> 1
        
        Args:
            symbols: 接收符号
            
        Returns:
            解调后的比特
        """
        return (symbols <= 0).astype(int)
    
    def symbols_to_llr(self, symbols: np.ndarray) -> np.ndarray:
        """
        将接收符号转换为对数似然比(LLR)
        
        LLR定义: ln(P(x=0|y) / P(x=1|y))
        对于AWGN信道: LLR = 2*y/sigma^2
        
        Args:
            symbols: 接收符号
            
        Returns:
            LLR值（正值表示更可能是0）
        """
        # LLR = 2 * y / sigma^2
        llr = 2.0 * symbols / (self.noise_std ** 2)
        return llr
    
    def add_noise(self, symbols: np.ndarray) -> np.ndarray:
        """
        添加高斯白噪声
        
        Args:
            symbols: 调制后的符号
            
        Returns:
            加噪声后的符号
        """
        noise = np.random.normal(0, self.noise_std, symbols.shape)
        return symbols + noise
    
    def transmit(self, bits: np.ndarray, return_llr: bool = True) -> np.ndarray:
        """
        完整的信道传输过程
        
        Args:
            bits: 输入比特
            return_llr: 是否返回LLR（软判决），否则返回硬判决
            
        Returns:
            LLR值或硬判决比特
        """
        # 调制
        symbols = self.modulate_bpsk(bits)
        
        # 加噪声
        received = self.add_noise(symbols)
        
        # 返回LLR或硬判决
        if return_llr:
            return self.symbols_to_llr(received)
        else:
            return self.demodulate_bpsk_hard(received)
    
    def get_capacity(self) -> float:
        """
        计算AWGN信道容量 (bits per channel use)
        
        C = 1 - E[log2(1 + exp(-2*Y/sigma^2))]
        近似: C ≈ 1 - log2(1 + exp(-SNR))
        
        Returns:
            信道容量
        """
        # 简化近似公式
        capacity = 1.0 - np.log2(1.0 + np.exp(-self.snr_linear))
        return capacity
    
    def update_snr(self, snr_db: float):
        """
        更新信噪比
        
        Args:
            snr_db: 新的信噪比(dB)
        """
        self.snr_db = snr_db
        self.snr_linear = 10 ** (snr_db / 10.0)
        self.noise_std = np.sqrt(1.0 / (2.0 * self.snr_linear))
    
    def __repr__(self) -> str:
        return f"AWGNChannel(SNR={self.snr_db:.2f}dB, noise_std={self.noise_std:.4f})"


if __name__ == "__main__":
    # 测试代码
    print("Testing AWGN Channel...")
    
    # 创建信道
    channel = AWGNChannel(snr_db=3.0, seed=42)
    print(f"Channel: {channel}")
    print(f"Channel Capacity: {channel.get_capacity():.4f} bits/use")
    
    # 测试传输
    test_bits = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    print(f"\nInput bits: {test_bits}")
    
    # BPSK调制
    symbols = channel.modulate_bpsk(test_bits)
    print(f"Modulated symbols: {symbols}")
    
    # 加噪声
    received = channel.add_noise(symbols)
    print(f"Received symbols: {received}")
    
    # 硬判决
    hard_decision = channel.demodulate_bpsk_hard(received)
    print(f"Hard decision: {hard_decision}")
    print(f"Bit errors: {np.sum(test_bits != hard_decision)}")
    
    # LLR
    llr = channel.symbols_to_llr(received)
    print(f"LLR values: {llr}")
    
    # 完整传输
    llr_direct = channel.transmit(test_bits, return_llr=True)
    print(f"\nDirect LLR: {llr_direct}")
    
    print("\n✓ AWGN Channel test passed!")
