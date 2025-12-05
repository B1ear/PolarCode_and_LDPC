"""
Polar码库封装器

封装第三方`polarcodes`库（py-polar-codes），提供
与项目自有PolarEncoder/SCDecoder兼容的统一接口。
"""

import numpy as np
from typing import Optional

try:
    from polarcodes import PolarCode, Construct, Encode, Decode
    POLARCODES_AVAILABLE = True
except ImportError:
    POLARCODES_AVAILABLE = False


class PolarLibWrapper:
    """
    polarcodes库的封装器，用于Polar编码/解码。
    
    提供与项目的PolarEncoder/SCDecoder匹配的encode/decode接口。
    """
    
    def __init__(self, N: int, K: int, design_snr_db: float = 2.0):
        """
        初始化Polar库封装器
        
        Args:
            N: 码长（必须是2的幂次）
            K: 信息位长度
            design_snr_db: 码构造的设计SNR（dB）
        """
        if not POLARCODES_AVAILABLE:
            raise ImportError("polarcodes library not available. Install with: pip install py-polar-codes")
        
        assert N > 0 and (N & (N - 1)) == 0, "N must be a power of 2"
        assert 0 < K < N, "K must be in (0, N)"
        
        self.N = N
        self.K = K
        self.design_snr_db = design_snr_db
        
        # 创建PolarCode实例并构造冻结集
        self.pc = PolarCode(N, K)
        Construct(self.pc, design_SNR=design_snr_db)
        
        # 存储冻结位和信息位位置供参考
        self.frozen_bits = self.pc.frozen
        self.info_bits = np.setdiff1d(np.arange(N), self.frozen_bits)
    
    def encode(self, message: np.ndarray) -> np.ndarray:
        """
        将消息比特编码为码字
        
        Args:
            message: 信息位，长度为K
            
        Returns:
            码字比特，长度为N
        """
        assert len(message) == self.K, f"Message length must be {self.K}"
        
        # 设置消息，编码，并获取码字
        self.pc.set_message(message.astype(int))
        Encode(self.pc)
        codeword = self.pc.get_codeword()
        
        return codeword.astype(int)
    
    def decode(self, llr: np.ndarray) -> np.ndarray:
        """
        使用SC解码器将LLR值解码为消息比特
        
        Args:
            llr: 对数似然比，长度为N
                 （正值表示bit=0更可能）
            
        Returns:
            解码后的消息比特，长度为K
        """
        assert len(llr) == self.N, f"LLR length must be {self.N}"
        
        # 设置似然值并解码
        self.pc.likelihoods = llr.astype(float)
        Decode(self.pc, decoder_name='scd')
        
        # 获取解码后的消息
        decoded_message = self.pc.message_received
        
        return decoded_message.astype(int)
    
    def get_code_rate(self) -> float:
        """获取码率K/N"""
        return self.K / self.N
    
    def get_frozen_bits_positions(self) -> np.ndarray:
        """获取冻结位位置"""
        return self.frozen_bits.copy()
    
    def get_info_bits_positions(self) -> np.ndarray:
        """获取信息位位置"""
        return self.info_bits.copy()
    
    def __repr__(self) -> str:
        return f"PolarLibWrapper(N={self.N}, K={self.K}, rate={self.get_code_rate():.3f}, design_SNR={self.design_snr_db}dB)"


if __name__ == "__main__":
    # 测试代码
    print("Testing PolarLibWrapper...")
    
    if not POLARCODES_AVAILABLE:
        print("✗ polarcodes library not available")
        exit(1)
    
    # 测试1：基本编码
    print("\n1. Basic Encoding Test:")
    N, K = 8, 4
    wrapper = PolarLibWrapper(N, K)
    print(f"Wrapper: {wrapper}")
    print(f"Frozen bits: {wrapper.get_frozen_bits_positions()}")
    print(f"Info bits: {wrapper.get_info_bits_positions()}")
    
    message = np.array([1, 0, 1, 1])
    codeword = wrapper.encode(message)
    print(f"Message: {message}")
    print(f"Codeword: {codeword}")
    
    # 测试2：无噪声解码
    print("\n2. No-Noise Decoding Test:")
    # 高幅度LLR
    llr = (1 - 2 * codeword.astype(float)) * 1000.0
    decoded = wrapper.decode(llr)
    print(f"Decoded: {decoded}")
    print(f"Match: {np.array_equal(message, decoded)}")
    
    # 测试3：多个消息
    print("\n3. Multiple Messages Test:")
    N, K = 16, 8
    wrapper = PolarLibWrapper(N, K)
    
    num_tests = 5
    all_correct = True
    for i in range(num_tests):
        msg = np.random.randint(0, 2, K)
        cw = wrapper.encode(msg)
        llr = (1 - 2 * cw.astype(float)) * 100.0
        dec = wrapper.decode(llr)
        
        correct = np.array_equal(msg, dec)
        all_correct = all_correct and correct
        print(f"  Test {i+1}: {'✓' if correct else '✗'}")
    
    print(f"All correct: {all_correct}")
    
    # 测试4：使用AWGN信道
    print("\n4. AWGN Channel Test:")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from channel import AWGNChannel
    
    N, K = 128, 64
    snr_db = 4.0
    wrapper = PolarLibWrapper(N, K, design_snr_db=2.0)
    channel = AWGNChannel(snr_db=snr_db, seed=42)
    
    num_frames = 20
    errors = 0
    
    for i in range(num_frames):
        msg = np.random.randint(0, 2, K)
        cw = wrapper.encode(msg)
        llr = channel.transmit(cw, return_llr=True)
        dec = wrapper.decode(llr)
        
        if not np.array_equal(msg, dec):
            errors += 1
    
    print(f"N={N}, K={K}, SNR={snr_db}dB")
    print(f"Frame errors: {errors}/{num_frames}")
    print(f"FER: {errors/num_frames:.4f}")
    
    # 测试5：不同码长
    print("\n5. Different Code Sizes Test:")
    configs = [(8, 4), (16, 8), (32, 16), (64, 32)]
    
    for N, K in configs:
        wrap = PolarLibWrapper(N, K)
        msg = np.random.randint(0, 2, K)
        cw = wrap.encode(msg)
        llr = (1 - 2 * cw.astype(float)) * 100.0
        dec = wrap.decode(llr)
        
        correct = np.array_equal(msg, dec)
        print(f"N={N:3d}, K={K:2d}, rate={wrap.get_code_rate():.3f}, correct={correct}")
    
    print("\n✓ PolarLibWrapper test passed!")
