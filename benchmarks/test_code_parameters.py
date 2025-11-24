"""
测试不同码长和码率的性能
Test performance across different code lengths and code rates
"""
import numpy as np
import sys
from pathlib import Path
import time
import json
import matplotlib.pyplot as plt

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polar import PolarEncoder, SCDecoder
from ldpc import LDPCEncoder, BPDecoder
from channel import AWGNChannel
from lib_wrappers import PolarLibWrapper, LDPCLibWrapper


def test_code_lengths(code_type='polar', rates=[0.5], snr_db=3.0, num_frames=50):
    """
    测试不同码长
    
    Args:
        code_type: 'polar' or 'ldpc'
        rates: 码率列表
        snr_db: 测试SNR
        num_frames: 测试帧数
    """
    if code_type == 'polar':
        # Polar码长必须是2的幂次 - 增加样本点
        code_lengths = [128, 256, 512, 1024, 2048, 4096]
    else:
        # LDPC码长 - 增加样本点
        code_lengths = [126, 252, 504, 1008, 2016, 4032]
    
    results = {
        'code_lengths': code_lengths,
        'rates': {},
        'snr_db': snr_db,
        'num_frames': num_frames
    }
    
    for rate in rates:
        print(f"\n{'='*70}")
        print(f"Testing {code_type.upper()} - Code Rate: {rate}")
        print(f"{'='*70}")
        
        rate_results = {
            'encoding_time': [],
            'decoding_time': [],
            'encoding_throughput': [],
            'decoding_throughput': [],
            'ber': [],
            'fer': []
        }
        
        for N in code_lengths:
            K = int(N * rate)
            
            print(f"\nN={N}, K={K}, rate={rate:.3f}")
            
            try:
                if code_type == 'polar':
                    # 使用第三方库的frozen bits
                    lib = PolarLibWrapper(N, K, 2.0)
                    frozen_bits = lib.get_frozen_bits_positions()
                    encoder = PolarEncoder(N, K, frozen_bits=frozen_bits)
                    decoder = SCDecoder(N, K, frozen_bits=frozen_bits)
                else:
                    # LDPC
                    lib = LDPCLibWrapper(N, K, dv=3, dc=6, seed=42)
                    K = lib.k  # 实际k
                    H = lib.get_parity_check_matrix()
                    G = lib.get_generator_matrix()
                    encoder = LDPCEncoder(N, K, H=H, G=G)
                    decoder = BPDecoder(H, max_iter=20)
                
                channel = AWGNChannel(snr_db)
                
                # 测试
                encode_times = []
                decode_times = []
                errors = 0
                frame_errors = 0
                
                for _ in range(num_frames):
                    msg = np.random.randint(0, 2, K)
                    
                    # 编码
                    t0 = time.time()
                    cw = encoder.encode(msg)
                    encode_times.append(time.time() - t0)
                    
                    # 信道
                    llr = channel.transmit(cw, return_llr=True)
                    
                    # 解码
                    t0 = time.time()
                    if code_type == 'ldpc':
                        decoded = decoder.decode(llr)
                        decoded_msg = decoded[:K]
                    else:
                        decoded_msg = decoder.decode(llr)
                    decode_times.append(time.time() - t0)
                    
                    # 统计错误
                    bit_errors = np.sum(msg != decoded_msg)
                    errors += bit_errors
                    if bit_errors > 0:
                        frame_errors += 1
                
                # 计算指标
                total_encode_time = np.sum(encode_times)
                total_decode_time = np.sum(decode_times)
                total_bits = num_frames * K
                
                encode_throughput = total_bits / total_encode_time / 1e6  # Mbps
                decode_throughput = total_bits / total_decode_time / 1e6
                ber = errors / total_bits
                fer = frame_errors / num_frames
                
                rate_results['encoding_time'].append(total_encode_time / num_frames)
                rate_results['decoding_time'].append(total_decode_time / num_frames)
                rate_results['encoding_throughput'].append(encode_throughput)
                rate_results['decoding_throughput'].append(decode_throughput)
                rate_results['ber'].append(ber)
                rate_results['fer'].append(fer)
                
                print(f"  Encoding: {total_encode_time/num_frames*1000:.2f}ms/frame, {encode_throughput:.4f} Mbps")
                print(f"  Decoding: {total_decode_time/num_frames*1000:.2f}ms/frame, {decode_throughput:.4f} Mbps")
                print(f"  BER: {ber:.6f}, FER: {fer:.4f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                rate_results['encoding_time'].append(None)
                rate_results['decoding_time'].append(None)
                rate_results['encoding_throughput'].append(None)
                rate_results['decoding_throughput'].append(None)
                rate_results['ber'].append(None)
                rate_results['fer'].append(None)
        
        results['rates'][rate] = rate_results
    
    return results


def test_code_rates(code_type='polar', N=1024, snr_db=3.0, num_frames=50):
    """
    测试不同码率
    
    Args:
        code_type: 'polar' or 'ldpc'
        N: 码长
        snr_db: 测试SNR
        num_frames: 测试帧数
    """
    # 常见码率 - 增加样本点密度
    rates = [1/4, 1/3, 2/5, 1/2, 3/5, 2/3, 3/4, 4/5, 5/6, 7/8]
    
    results = {
        'N': N,
        'rates': rates,
        'snr_db': snr_db,
        'num_frames': num_frames,
        'K_values': [],
        'encoding_time': [],
        'decoding_time': [],
        'encoding_throughput': [],
        'decoding_throughput': [],
        'ber': [],
        'fer': []
    }
    
    print(f"\n{'='*70}")
    print(f"Testing {code_type.upper()} - Code Length: N={N}")
    print(f"{'='*70}")
    
    for rate in rates:
        K = int(N * rate)
        results['K_values'].append(K)
        
        print(f"\nRate={rate:.3f} (N={N}, K={K})")
        
        try:
            if code_type == 'polar':
                lib = PolarLibWrapper(N, K, 2.0)
                frozen_bits = lib.get_frozen_bits_positions()
                encoder = PolarEncoder(N, K, frozen_bits=frozen_bits)
                decoder = SCDecoder(N, K, frozen_bits=frozen_bits)
            else:
                lib = LDPCLibWrapper(N, K, dv=3, dc=6, seed=42)
                K = lib.k
                results['K_values'][-1] = K  # 更新实际K
                H = lib.get_parity_check_matrix()
                G = lib.get_generator_matrix()
                encoder = LDPCEncoder(N, K, H=H, G=G)
                decoder = BPDecoder(H, max_iter=20)
            
            channel = AWGNChannel(snr_db)
            
            # 测试
            encode_times = []
            decode_times = []
            errors = 0
            frame_errors = 0
            
            for _ in range(num_frames):
                msg = np.random.randint(0, 2, K)
                
                t0 = time.time()
                cw = encoder.encode(msg)
                encode_times.append(time.time() - t0)
                
                llr = channel.transmit(cw, return_llr=True)
                
                t0 = time.time()
                if code_type == 'ldpc':
                    decoded = decoder.decode(llr)
                    decoded_msg = decoded[:K]
                else:
                    decoded_msg = decoder.decode(llr)
                decode_times.append(time.time() - t0)
                
                bit_errors = np.sum(msg != decoded_msg)
                errors += bit_errors
                if bit_errors > 0:
                    frame_errors += 1
            
            total_encode_time = np.sum(encode_times)
            total_decode_time = np.sum(decode_times)
            total_bits = num_frames * K
            
            encode_throughput = total_bits / total_encode_time / 1e6
            decode_throughput = total_bits / total_decode_time / 1e6
            ber = errors / total_bits
            fer = frame_errors / num_frames
            
            results['encoding_time'].append(total_encode_time / num_frames)
            results['decoding_time'].append(total_decode_time / num_frames)
            results['encoding_throughput'].append(encode_throughput)
            results['decoding_throughput'].append(decode_throughput)
            results['ber'].append(ber)
            results['fer'].append(fer)
            
            print(f"  Encoding: {total_encode_time/num_frames*1000:.2f}ms/frame, {encode_throughput:.4f} Mbps")
            print(f"  Decoding: {total_decode_time/num_frames*1000:.2f}ms/frame, {decode_throughput:.4f} Mbps")
            print(f"  BER: {ber:.6f}, FER: {fer:.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results['encoding_time'].append(None)
            results['decoding_time'].append(None)
            results['encoding_throughput'].append(None)
            results['decoding_throughput'].append(None)
            results['ber'].append(None)
            results['fer'].append(None)
    
    return results


def plot_results(length_results, rate_results, output_dir='results/code_params'):
    """绘制结果图表"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. 码长 vs 吞吐量
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for code_type in ['polar', 'ldpc']:
        if code_type in length_results:
            data = length_results[code_type]
            for rate, results in data['rates'].items():
                lengths = data['code_lengths']
                throughput = results['decoding_throughput']
                # 过滤None值
                valid = [(l, t) for l, t in zip(lengths, throughput) if t is not None]
                if valid:
                    lengths_valid, throughput_valid = zip(*valid)
                    ax1.plot(lengths_valid, throughput_valid, 'o-', 
                            label=f'{code_type.upper()} (rate={rate})', linewidth=2)
    
    ax1.set_xlabel('Code Length N', fontsize=12)
    ax1.set_ylabel('Decoding Throughput (Mbps)', fontsize=12)
    ax1.set_title('Throughput vs Code Length', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # 2. 码率 vs 吞吐量
    for code_type in ['polar', 'ldpc']:
        if code_type in rate_results:
            data = rate_results[code_type]
            rates = data['rates']
            throughput = data['decoding_throughput']
            valid = [(r, t) for r, t in zip(rates, throughput) if t is not None]
            if valid:
                rates_valid, throughput_valid = zip(*valid)
                ax2.plot(rates_valid, throughput_valid, 'o-',
                        label=f'{code_type.upper()} (N={data["N"]})', linewidth=2)
    
    ax2.set_xlabel('Code Rate', fontsize=12)
    ax2.set_ylabel('Decoding Throughput (Mbps)', fontsize=12)
    ax2.set_title('Throughput vs Code Rate', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/throughput_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/throughput_comparison.png")
    plt.close()
    
    # 3. BER性能
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for code_type in ['polar', 'ldpc']:
        if code_type in rate_results:
            data = rate_results[code_type]
            rates = data['rates']
            ber = data['ber']
            valid = [(r, b) for r, b in zip(rates, ber) if b is not None]
            if valid:
                rates_valid, ber_valid = zip(*valid)
                ber_valid = [max(b, 1e-6) for b in ber_valid]  # 避免log(0)
                ax1.semilogy(rates_valid, ber_valid, 'o-',
                            label=f'{code_type.upper()} (N={data["N"]})', linewidth=2)
    
    ax1.set_xlabel('Code Rate', fontsize=12)
    ax1.set_ylabel('BER', fontsize=12)
    ax1.set_title('BER vs Code Rate', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 复杂度 (操作数)
    for code_type in ['polar', 'ldpc']:
        if code_type in length_results:
            data = length_results[code_type]
            for rate, results in data['rates'].items():
                lengths = data['code_lengths']
                # 估算复杂度
                if code_type == 'polar':
                    ops = [N * np.log2(N) for N in lengths]
                else:
                    ops = [N * 3 * 20 for N in lengths]  # n*dv*iter
                
                ax2.loglog(lengths, ops, 'o-',
                          label=f'{code_type.upper()} (rate={rate})', linewidth=2, base=2)
    
    ax2.set_xlabel('Code Length N', fontsize=12)
    ax2.set_ylabel('Operations', fontsize=12)
    ax2.set_title('Complexity vs Code Length', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ber_complexity_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/ber_complexity_comparison.png")
    plt.close()


if __name__ == '__main__':
    print("="*70)
    print("Code Parameters Performance Test")
    print("="*70)
    
    # 测试不同码长
    print("\n\n" + "="*70)
    print("PART 1: Testing Different Code Lengths")
    print("="*70)
    
    polar_length_results = test_code_lengths('polar', rates=[0.5], snr_db=3.0, num_frames=50)
    ldpc_length_results = test_code_lengths('ldpc', rates=[0.5], snr_db=3.0, num_frames=50)
    
    # 测试不同码率 - 使用相近N进行对比
    print("\n\n" + "="*70)
    print("PART 2: Testing Different Code Rates (Similar N for Comparison)")
    print("="*70)
    
    # Polar使用N=1024, LDPC使用N=1008 (dc=6能整除1008)
    polar_rate_results = test_code_rates('polar', N=1024, snr_db=3.0, num_frames=50)
    ldpc_rate_results = test_code_rates('ldpc', N=1008, snr_db=3.0, num_frames=50)
    
    # 保存结果
    output_dir = Path('results/code_params')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'length_tests': {
            'polar': polar_length_results,
            'ldpc': ldpc_length_results
        },
        'rate_tests': {
            'polar': polar_rate_results,
            'ldpc': ldpc_rate_results
        }
    }
    
    with open(output_dir / 'code_params_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_dir}/code_params_results.json")
    
    # 绘图
    plot_results(
        results['length_tests'],
        results['rate_tests'],
        output_dir=str(output_dir)
    )
    
    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)
