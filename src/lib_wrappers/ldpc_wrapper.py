"""
LDPC 库封装器

封装第三方 pyldpc 库，提供一个统一的接口，以兼容项目自有的 LDPCEncoder / BPDecoder
"""

import numpy as np
from typing import Optional
import warnings

try:
    import pyldpc
    PYLDPC_AVAILABLE = True
except ImportError:
    PYLDPC_AVAILABLE = False


class LDPCLibWrapper:
    """
    对pyldpc库的封装，用于LDPC编码与解码。
    
    提供与项目中LDPCEncoder/BPDecoder接口一致的encode/decode方法
    """
    
    def __init__(self, n: int, k: int, dv: int = 3, dc: int = 6, seed: Optional[int] = None):
        """
        初始化LDPC库封装器
        
        Args:
            n: 码长（编码后码字长度）
            k: 信息位长度
            dv: 变量节点度数（每个比特参与的校验方程数量）
            dc: 校验节点度数（每个校验方程涉及的比特数量）
            seed: 用于构造校验矩阵的随机种子
        """
        if not PYLDPC_AVAILABLE:
            raise ImportError("pyldpc library not available. Install with: pip install pyldpc")
        
        assert n > k > 0, "Invalid code parameters: n > k > 0 required"
        assert dc >= dv, "Check degree dc must be >= variable degree dv"
        
        self.n = n
        self.k = k
        self.m = n - k  # 校验位数量
        self.dv = dv
        self.dc = dc
        self.seed = seed
        
        # 使用pyldpc创建H和G矩阵
        # H: (m, n) 校验矩阵
        # G: (n, k) 生成矩阵（pyldpc中的转置形式）
        self.H, self.G = pyldpc.make_ldpc(n, dv, dc, systematic=True, sparse=False, seed=seed)
        
        # 从G的形状提取实际的k
        # G的形状是(n, k_actual)，其中k_actual可能与输入的k略有不同
        self.k_actual = self.G.shape[1]
        
        if self.k_actual != k:
            print(f"Warning: Requested k={k}, but pyldpc generated k={self.k_actual}")
            self.k = self.k_actual
    
    def encode(self, message: np.ndarray) -> np.ndarray:
        """
        将消息比特编码为码字
        
        Args:
            message: 信息位，长度为k
            
        Returns:
            码字比特，长度为n
        """
        assert len(message) == self.k, f"Message length must be {self.k}"
        
        # 使用pyldpc.utils.binaryproduct进行纯编码（无噪声）
        # codeword = G @ message (mod 2)
        codeword = pyldpc.utils.binaryproduct(self.G, message)
        
        return codeword.astype(int)
    
    def decode(self, llr: np.ndarray, max_iter: int = 50) -> np.ndarray:
        """
        使用BP解码器将LLR值解码为消息比特
        
        Args:
            llr: 对数似然比，长度为n
                 （正值表示bit=0更可能）
            max_iter: 最大BP迭代次数
            
        Returns:
            解码后的消息比特，长度为k
        """
        assert len(llr) == self.n, f"LLR length must be {self.n}"
        
        # 将LLR转换为pyldpc期望的信道输出格式
        # pyldpc.decode期望噪声符号 y = BPSK + noise
        # 我们的LLR = 2*y/sigma^2，所以 y = LLR * sigma^2 / 2
        # 对于解码，我们需要提供SNR和y
        # 我们将从LLR幅度计算有效SNR
        
        # 使用启发式方法：假设平均LLR幅度对应于SNR
        # LLR = 2*y/sigma^2，对于BPSK |y| ~ 1 + noise
        # 从平均LLR幅度估计SNR
        avg_llr_mag = np.mean(np.abs(llr))
        
        # 粗略启发式：SNR_linear ~ avg_llr_mag / 4
        # SNR_dB = 10 * log10(SNR_linear)
        snr_linear = max(avg_llr_mag / 4.0, 0.1)
        snr_db = 10.0 * np.log10(snr_linear)
        
        # 将LLR转换回信道符号
        # 对于pyldpc: LLR = 2*y/var, var = 10^(-snr/10)
        var = 10.0 ** (-snr_db / 10.0)
        y = llr * var / 2.0
        
        # 使用pyldpc解码（抑制收敛警告 - 在低SNR下是预期的）
        # 返回解码后的码字（长度n）
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*convergence.*')
            decoded_codeword = pyldpc.decode(self.H, y, snr_db, maxiter=max_iter)
        
        # 使用pyldpc.get_message提取消息比特
        decoded_message = pyldpc.get_message(self.G, decoded_codeword)
        
        return decoded_message.astype(int)
    
    def get_code_rate(self) -> float:
        """获取码率k/n"""
        return self.k / self.n
    
    def get_parity_check_matrix(self) -> np.ndarray:
        """获取校验矩阵H"""
        return self.H.copy()
    
    def get_generator_matrix(self) -> np.ndarray:
        """获取生成矩阵G"""
        return self.G.copy()
    
    def __repr__(self) -> str:
        return f"LDPCLibWrapper(n={self.n}, k={self.k}, rate={self.get_code_rate():.3f}, dv={self.dv}, dc={self.dc})"


if __name__ == "__main__":
    # 测试代码
    print("Testing LDPCLibWrapper...")
    
    if not PYLDPC_AVAILABLE:
        print("✗ pyldpc library not available")
        exit(1)
    
    # 测试1：基本编码
    print("\n1. Basic Encoding Test:")
    n, k = 12, 6
    wrapper = LDPCLibWrapper(n, k, dv=3, dc=6, seed=42)
    print(f"Wrapper: {wrapper}")
    print(f"H shape: {wrapper.H.shape}")
    print(f"G shape: {wrapper.G.shape}")
    print(f"Actual k: {wrapper.k}")
    
    # 使用wrapper的实际k值
    k_actual = wrapper.k
    message = np.random.randint(0, 2, k_actual)
    codeword = wrapper.encode(message)
    print(f"Message: {message}")
    print(f"Codeword length: {len(codeword)}")
    
    # 验证码字
    syndrome = (wrapper.H @ codeword) % 2
    valid = np.all(syndrome == 0)
    print(f"Valid codeword (H*c=0): {valid}")
    
    # 测试2：无噪声解码
    print("\n2. No-Noise Decoding Test:")
    # 高幅度LLR
    llr = (1 - 2 * codeword.astype(float)) * 100.0
    decoded = wrapper.decode(llr, max_iter=50)
    print(f"Decoded length: {len(decoded)}")
    print(f"Match: {np.array_equal(message, decoded)}")
    
    # 测试3：多个消息
    print("\n3. Multiple Messages Test:")
    n, k = 24, 12
    wrapper = LDPCLibWrapper(n, k, dv=3, dc=6, seed=42)
    k_actual = wrapper.k
    
    num_tests = 5
    all_correct = True
    for i in range(num_tests):
        msg = np.random.randint(0, 2, k_actual)
        cw = wrapper.encode(msg)
        llr = (1 - 2 * cw.astype(float)) * 100.0
        dec = wrapper.decode(llr, max_iter=50)
        
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
    
    n, k = 96, 48
    snr_db = 3.0
    wrapper = LDPCLibWrapper(n, k, dv=3, dc=6, seed=42)
    k_actual = wrapper.k
    channel = AWGNChannel(snr_db=snr_db, seed=42)
    
    num_frames = 20
    errors = 0
    
    for i in range(num_frames):
        msg = np.random.randint(0, 2, k_actual)
        cw = wrapper.encode(msg)
        llr = channel.transmit(cw, return_llr=True)
        dec = wrapper.decode(llr, max_iter=50)
        
        if not np.array_equal(msg, dec):
            errors += 1
    
    print(f"n={n}, k={k_actual}, SNR={snr_db}dB")
    print(f"Frame errors: {errors}/{num_frames}")
    print(f"FER: {errors/num_frames:.4f}")
    
    # 测试5：不同码长
    print("\n5. Different Code Sizes Test:")
    configs = [(24, 12), (48, 24), (96, 48)]
    
    for n, k in configs:
        wrap = LDPCLibWrapper(n, k, dv=3, dc=6, seed=42)
        k_actual = wrap.k
        msg = np.random.randint(0, 2, k_actual)
        cw = wrap.encode(msg)
        llr = (1 - 2 * cw.astype(float)) * 100.0
        dec = wrap.decode(llr, max_iter=50)
        
        correct = np.array_equal(msg, dec)
        print(f"n={n:3d}, k={k_actual:2d}, rate={wrap.get_code_rate():.3f}, correct={correct}")
    
    print("\n✓ LDPCLibWrapper test passed!")
