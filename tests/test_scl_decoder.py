"""
SCL解码器实现的测试脚本。

比较SC和SCL解码器性能。
"""

import numpy as np
from src.polar.encoder import PolarEncoder
from src.polar.decoder import SCDecoder, SCLDecoder
from src.channel.awgn import AWGNChannel


def test_scl_basic():
    """测试基本的SCL解码器功能。"""
    print("=== Basic SCL Decoder Test ===\n")
    
    N, K = 16, 8
    list_sizes = [1, 2, 4, 8]
    
    # 创建编码器
    encoder = PolarEncoder(N, K)
    
    # 生成随机消息
    np.random.seed(42)
    message = np.random.randint(0, 2, K)
    print(f"Original message: {message}")
    
    # 编码
    codeword = encoder.encode(message)
    print(f"Codeword: {codeword}")
    
    # 添加噪声
    snr_db = 2.0
    channel = AWGNChannel(snr_db)
    llr = channel.transmit(codeword, return_llr=True)
    
    print(f"\nSNR: {snr_db} dB")
    print(f"Received (first 10 LLRs): {llr[:10]}")
    
    # 测试不同的列表大小
    print("\n--- Decoding Results ---")
    for L in list_sizes:
        decoder = SCLDecoder(N, K, list_size=L, frozen_bits=encoder.frozen_bits)
        decoded = decoder.decode(llr)
        errors = np.sum(decoded != message)
        print(f"L={L}: decoded={decoded}, errors={errors}")
    
    print()


def test_scl_vs_sc():
    """比较SCL与SC解码器的误码率"""
    print("=== SCL vs SC Performance Comparison ===\n")
    
    N, K = 64, 32
    n_trials = 100
    snr_db = 1.0
    
    # Create encoder
    encoder = PolarEncoder(N, K)
    
    # Create decoders
    sc_decoder = SCDecoder(N, K, frozen_bits=encoder.frozen_bits)
    scl_decoders = {
        L: SCLDecoder(N, K, list_size=L, frozen_bits=encoder.frozen_bits)
        for L in [1, 2, 4, 8]
    }
    
    # Channel
    channel = AWGNChannel(snr_db)
    
    # Count errors
    sc_errors = 0
    scl_errors = {L: 0 for L in scl_decoders.keys()}
    
    np.random.seed(123)
    
    print(f"Running {n_trials} trials at SNR={snr_db} dB...")
    print(f"N={N}, K={K}\n")
    
    for trial in range(n_trials):
        # Generate and encode
        message = np.random.randint(0, 2, K)
        codeword = encoder.encode(message)
        
        # Transmit
        llr = channel.transmit(codeword, return_llr=True)
        
        # SC decode
        decoded_sc = sc_decoder.decode(llr)
        if not np.array_equal(decoded_sc, message):
            sc_errors += 1
        
        # SCL decode
        for L, decoder in scl_decoders.items():
            decoded_scl = decoder.decode(llr)
            if not np.array_equal(decoded_scl, message):
                scl_errors[L] += 1
        
        if (trial + 1) % 20 == 0:
            print(f"  Completed {trial + 1}/{n_trials} trials...")
    
    # Print results
    print("\n--- Results ---")
    print(f"SC Decoder:  {sc_errors}/{n_trials} errors ({sc_errors/n_trials*100:.2f}%)")
    for L in sorted(scl_errors.keys()):
        errors = scl_errors[L]
        print(f"SCL (L={L}):  {errors}/{n_trials} errors ({errors/n_trials*100:.2f}%)")
    
    print()


def test_scl_edge_cases():
    """Test SCL decoder edge cases."""
    print("=== SCL Edge Cases Test ===\n")
    
    # Test 1: L=1 should behave like SC
    print("Test 1: L=1 vs SC decoder")
    N, K = 32, 16
    encoder = PolarEncoder(N, K)
    
    np.random.seed(999)
    message = np.random.randint(0, 2, K)
    codeword = encoder.encode(message)
    
    channel = AWGNChannel(2.0)
    llr = channel.transmit(codeword, return_llr=True)
    
    sc_decoder = SCDecoder(N, K, frozen_bits=encoder.frozen_bits)
    scl_decoder = SCLDecoder(N, K, list_size=1, frozen_bits=encoder.frozen_bits)
    
    decoded_sc = sc_decoder.decode(llr)
    decoded_scl = scl_decoder.decode(llr)
    
    print(f"  SC result:  {decoded_sc}")
    print(f"  SCL result: {decoded_scl}")
    print(f"  Match: {np.array_equal(decoded_sc, decoded_scl)}")
    
    # Test 2: All zeros
    print("\nTest 2: All-zero codeword")
    codeword_zero = np.zeros(N, dtype=int)
    llr_zero = channel.transmit(codeword_zero, return_llr=True)
    
    decoded_zero = scl_decoder.decode(llr_zero)
    print(f"  Decoded: {decoded_zero}")
    print(f"  All zeros: {np.all(decoded_zero == 0)}")
    
    # Test 3: High SNR (should decode perfectly)
    print("\nTest 3: High SNR (10 dB)")
    high_snr_channel = AWGNChannel(10.0)
    llr_high = high_snr_channel.transmit(codeword, return_llr=True)
    
    decoded_high = scl_decoder.decode(llr_high)
    print(f"  Original:  {message}")
    print(f"  Decoded:   {decoded_high}")
    print(f"  Perfect:   {np.array_equal(decoded_high, message)}")
    
    print()


if __name__ == "__main__":
    test_scl_basic()
    test_scl_vs_sc()
    test_scl_edge_cases()
    
    print("=== All Tests Completed ===")
