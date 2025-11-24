"""
Fading Channel Implementation

衰落信道
"""

import numpy as np


class RayleighFadingChannel:
    """
    Rayleigh衰落信道
    """
    
    def __init__(self, snr_db: float, seed: int = None):
        """
        初始化Rayleigh衰落信道
        
        Args:
            snr_db: 平均信噪比(dB)
            seed: 随机种子
        """
        self.snr_db = snr_db
        self.snr_linear = 10 ** (snr_db / 10.0)
        self.noise_std = np.sqrt(1.0 / (2.0 * self.snr_linear))
        
        if seed is not None:
            np.random.seed(seed)
    
    def transmit(self, bits: np.ndarray, return_llr: bool = True) -> np.ndarray:
        """
        通过Rayleigh衰落信道传输
        
        Args:
            bits: 输入比特
            return_llr: 是否返回LLR
            
        Returns:
            LLR值或硬判决比特
        """
        # BPSK调制
        symbols = 1.0 - 2.0 * bits.astype(float)
        
        # Rayleigh衰落系数 (复数)
        h_real = np.random.normal(0, 1/np.sqrt(2), len(symbols))
        h_imag = np.random.normal(0, 1/np.sqrt(2), len(symbols))
        h = h_real + 1j * h_imag
        h_mag = np.abs(h)
        
        # 衰落后的信号
        faded_symbols = h_mag * symbols
        
        # 添加噪声
        noise = np.random.normal(0, self.noise_std, len(symbols))
        received = faded_symbols + noise
        
        if return_llr:
            # 计算LLR（考虑信道状态信息）
            llr = 2.0 * received * h_mag / (self.noise_std ** 2)
            return llr
        else:
            # 硬判决
            return (received <= 0).astype(int)
    
    def __repr__(self) -> str:
        return f"RayleighFadingChannel(SNR={self.snr_db:.2f}dB)"
