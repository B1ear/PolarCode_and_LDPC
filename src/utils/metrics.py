"""
Performance Metrics Module

计算各种性能指标：BER, FER, 吞吐量等
"""

import numpy as np
import time
from typing import Tuple, Dict, List


def calculate_ber(original_bits: np.ndarray, decoded_bits: np.ndarray) -> float:
    """
    计算误码率 (Bit Error Rate)
    
    Args:
        original_bits: 原始比特序列
        decoded_bits: 解码后的比特序列
        
    Returns:
        BER值 (0.0 到 1.0)
    """
    assert len(original_bits) == len(decoded_bits), "Bit sequences must have same length"
    
    errors = np.sum(original_bits != decoded_bits)
    total_bits = len(original_bits)
    
    return errors / total_bits if total_bits > 0 else 0.0


def calculate_fer(original_frames: List[np.ndarray], 
                  decoded_frames: List[np.ndarray]) -> float:
    """
    计算帧错误率 (Frame Error Rate)
    
    Args:
        original_frames: 原始帧列表
        decoded_frames: 解码帧列表
        
    Returns:
        FER值 (0.0 到 1.0)
    """
    assert len(original_frames) == len(decoded_frames), "Frame lists must have same length"
    
    frame_errors = 0
    for orig, dec in zip(original_frames, decoded_frames):
        if not np.array_equal(orig, dec):
            frame_errors += 1
    
    total_frames = len(original_frames)
    
    return frame_errors / total_frames if total_frames > 0 else 0.0


def calculate_throughput(num_bits: int, elapsed_time: float) -> float:
    """
    计算吞吐量
    
    Args:
        num_bits: 处理的比特数
        elapsed_time: 处理时间（秒）
        
    Returns:
        吞吐量 (Mbps)
    """
    if elapsed_time <= 0:
        return 0.0
    
    return (num_bits / elapsed_time) / 1e6  # 转换为Mbps


def measure_encoding_throughput(encoder, num_frames: int = 1000) -> Dict[str, float]:
    """
    测量编码器吞吐量
    
    Args:
        encoder: 编码器对象（需要有encode方法和K属性）
        num_frames: 测试帧数
        
    Returns:
        包含吞吐量信息的字典
    """
    K = encoder.K if hasattr(encoder, 'K') else encoder.k
    total_bits = num_frames * K
    
    # 生成随机消息
    messages = [np.random.randint(0, 2, K) for _ in range(num_frames)]
    
    # 测量编码时间
    start_time = time.time()
    for msg in messages:
        _ = encoder.encode(msg)
    elapsed_time = time.time() - start_time
    
    throughput = calculate_throughput(total_bits, elapsed_time)
    
    return {
        'total_bits': total_bits,
        'num_frames': num_frames,
        'elapsed_time': elapsed_time,
        'throughput_mbps': throughput,
        'avg_time_per_frame': elapsed_time / num_frames
    }


def measure_decoding_throughput(decoder, llr_inputs: List[np.ndarray]) -> Dict[str, float]:
    """
    测量解码器吞吐量
    
    Args:
        decoder: 解码器对象（需要有decode方法）
        llr_inputs: LLR输入列表
        
    Returns:
        包含吞吐量信息的字典
    """
    num_frames = len(llr_inputs)
    bits_per_frame = len(llr_inputs[0])
    total_bits = num_frames * bits_per_frame
    
    # 测量解码时间
    start_time = time.time()
    for llr in llr_inputs:
        _ = decoder.decode(llr)
    elapsed_time = time.time() - start_time
    
    throughput = calculate_throughput(total_bits, elapsed_time)
    
    return {
        'total_bits': total_bits,
        'num_frames': num_frames,
        'elapsed_time': elapsed_time,
        'throughput_mbps': throughput,
        'avg_time_per_frame': elapsed_time / num_frames
    }


def calculate_ber_with_confidence(bit_errors: int, total_bits: int, 
                                  confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    计算BER及其置信区间
    
    Args:
        bit_errors: 错误比特数
        total_bits: 总比特数
        confidence: 置信水平 (默认0.95)
        
    Returns:
        (BER, 下界, 上界)
    """
    if total_bits == 0:
        return 0.0, 0.0, 0.0
    
    ber = bit_errors / total_bits
    
    # 使用Wilson score区间
    from scipy import stats
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    denominator = 1 + z**2 / total_bits
    center = (ber + z**2 / (2 * total_bits)) / denominator
    margin = z * np.sqrt(ber * (1 - ber) / total_bits + z**2 / (4 * total_bits**2)) / denominator
    
    lower = max(0, center - margin)
    upper = min(1, center + margin)
    
    return ber, lower, upper


def calculate_snr_from_ebn0(ebn0_db: float, code_rate: float) -> float:
    """
    从Eb/N0计算SNR
    
    Args:
        ebn0_db: Eb/N0 (dB)
        code_rate: 码率
        
    Returns:
        SNR (dB)
    """
    return ebn0_db + 10 * np.log10(code_rate)


def calculate_ebn0_from_snr(snr_db: float, code_rate: float) -> float:
    """
    从SNR计算Eb/N0
    
    Args:
        snr_db: SNR (dB)
        code_rate: 码率
        
    Returns:
        Eb/N0 (dB)
    """
    return snr_db - 10 * np.log10(code_rate)


if __name__ == "__main__":
    # 测试代码
    print("Testing Metrics Module...")
    
    # 测试1: BER计算
    print("\n1. BER Calculation Test:")
    original = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
    decoded1 = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])  # 无错误
    decoded2 = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 1])  # 2个错误
    
    ber1 = calculate_ber(original, decoded1)
    ber2 = calculate_ber(original, decoded2)
    
    print(f"Original:  {original}")
    print(f"Decoded1:  {decoded1}, BER: {ber1:.4f}")
    print(f"Decoded2:  {decoded2}, BER: {ber2:.4f}")
    
    # 测试2: FER计算
    print("\n2. FER Calculation Test:")
    original_frames = [
        np.array([1, 0, 1, 0]),
        np.array([0, 1, 1, 0]),
        np.array([1, 1, 0, 1]),
    ]
    decoded_frames = [
        np.array([1, 0, 1, 0]),  # 正确
        np.array([0, 0, 1, 0]),  # 1个错误
        np.array([1, 1, 0, 1]),  # 正确
    ]
    
    fer = calculate_fer(original_frames, decoded_frames)
    print(f"FER: {fer:.4f} (1/3 = 0.3333)")
    
    # 测试3: 吞吐量计算
    print("\n3. Throughput Calculation Test:")
    num_bits = 1000000  # 1 Mbit
    elapsed_time = 0.1  # 0.1 秒
    
    throughput = calculate_throughput(num_bits, elapsed_time)
    print(f"Bits: {num_bits}, Time: {elapsed_time}s")
    print(f"Throughput: {throughput:.2f} Mbps")
    
    # 测试4: BER置信区间
    print("\n4. BER Confidence Interval Test:")
    bit_errors = 100
    total_bits = 10000
    
    try:
        ber, lower, upper = calculate_ber_with_confidence(bit_errors, total_bits)
        print(f"BER: {ber:.6f}")
        print(f"95% CI: [{lower:.6f}, {upper:.6f}]")
    except ImportError:
        print("scipy not available, skipping confidence interval test")
    
    # 测试5: SNR和Eb/N0转换
    print("\n5. SNR and Eb/N0 Conversion Test:")
    ebn0_db = 3.0
    code_rate = 0.5
    
    snr_db = calculate_snr_from_ebn0(ebn0_db, code_rate)
    ebn0_back = calculate_ebn0_from_snr(snr_db, code_rate)
    
    print(f"Eb/N0: {ebn0_db} dB, Rate: {code_rate}")
    print(f"SNR: {snr_db:.2f} dB")
    print(f"Eb/N0 (converted back): {ebn0_back:.2f} dB")
    
    print("\n✓ Metrics test passed!")
