"""
SNR性能曲线测试
测试不同SNR下的BER/FER性能曲线

测试Polar码和LDPC码在多种码率下的性能，
生成标准性能曲线。
"""
import numpy as np
import sys
from pathlib import Path
import time
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polar import PolarEncoder, SCDecoder
from ldpc import LDPCEncoder, BPDecoder
from channel import AWGNChannel
from lib_wrappers import PolarLibWrapper, LDPCLibWrapper


def simulate_snr_curve(
    code_type: str,
    N: int,
    K: int,
    snr_range: np.ndarray,
    num_frames: int = 100,
    max_errors: int = 100,
    use_library: bool = False
) -> Tuple[List[float], List[float], Dict]:
    """
    测试单个配置下的SNR曲线
    
    Args:
        code_type: 'polar' or 'ldpc'
        N: 码长
        K: 信息位长度
        snr_range: SNR范围 (dB)
        num_frames: 每个SNR点测试的最大帧数
        max_errors: 达到此错误帧数后提前停止
        use_library: 是否使用第三方库实现
        
    Returns:
        ber_list: BER列表
        fer_list: FER列表
        stats: 统计信息字典
    """
    print(f"\n{'='*70}")
    print(f"Testing {code_type.upper()}: N={N}, K={K}, rate={K/N:.3f}")
    print(f"Implementation: {'Library' if use_library else 'Self'}")
    print(f"{'='*70}")
    
    # 初始化编码器和解码器
    if code_type == 'polar':
        lib = PolarLibWrapper(N, K, 2.0)
        frozen_bits = lib.get_frozen_bits_positions()
        
        if use_library:
            encoder = lib
            decoder = lib
        else:
            encoder = PolarEncoder(N, K, frozen_bits=frozen_bits)
            decoder = SCDecoder(N, K, frozen_bits=frozen_bits)
    else:  # ldpc
        lib = LDPCLibWrapper(N, K, dv=3, dc=6, seed=42)
        K = lib.k  # 更新实际K值
        
        if use_library:
            encoder = lib
            decoder = lib
        else:
            H = lib.get_parity_check_matrix()
            G = lib.get_generator_matrix()
            encoder = LDPCEncoder(N, K, H=H, G=G)
            decoder = BPDecoder(H, max_iter=20)
    
    ber_list = []
    fer_list = []
    stats = {
        'frames_tested': [],
        'total_bits': [],
        'error_bits': [],
        'frame_errors': [],
        'simulation_time': []
    }
    
    for snr_db in snr_range:
        print(f"\nSNR = {snr_db:.1f} dB", end=" ")
        
        channel = AWGNChannel(snr_db=snr_db)
        
        total_bits = 0
        error_bits = 0
        frame_errors = 0
        frames_tested = 0
        
        start_time = time.time()
        
        for _ in range(num_frames):
            # 生成随机消息
            message = np.random.randint(0, 2, K)
            
            # 编码
            if use_library:
                if code_type == 'polar':
                    codeword = encoder.encode(message)
                else:
                    codeword = encoder.encode(message)
            else:
                codeword = encoder.encode(message)
            
            # 信道传输
            llr = channel.transmit(codeword, return_llr=True)
            
            # 解码
            if use_library:
                if code_type == 'polar':
                    decoded = decoder.decode(llr)
                else:
                    decoded = decoder.decode(llr)
                    decoded = decoded[:K]  # LDPC返回完整码字，取前K位
            else:
                if code_type == 'ldpc':
                    decoded_full = decoder.decode(llr)
                    decoded = decoded_full[:K]
                else:
                    decoded = decoder.decode(llr)
            
            # 计算错误
            bit_errors = np.sum(message != decoded)
            total_bits += K
            error_bits += bit_errors
            
            if bit_errors > 0:
                frame_errors += 1
            
            frames_tested += 1
            
            # 提前停止条件
            if frame_errors >= max_errors:
                break
        
        elapsed = time.time() - start_time
        
        # 计算BER和FER
        ber = error_bits / total_bits if total_bits > 0 else 0
        fer = frame_errors / frames_tested if frames_tested > 0 else 0
        
        ber_list.append(ber)
        fer_list.append(fer)
        
        stats['frames_tested'].append(frames_tested)
        stats['total_bits'].append(total_bits)
        stats['error_bits'].append(error_bits)
        stats['frame_errors'].append(frame_errors)
        stats['simulation_time'].append(elapsed)
        
        print(f"-> BER: {ber:.6f}, FER: {fer:.4f} ({frames_tested} frames, {elapsed:.1f}s)")
    
    return ber_list, fer_list, stats


def test_multiple_rates(
    code_type: str,
    N_base: int,
    rates: List[float],
    snr_range: np.ndarray,
    num_frames: int = 100,
    max_errors: int = 100,
    test_library: bool = True
) -> Dict:
    """
    测试多个码率下的SNR曲线
    
    Args:
        code_type: 'polar' or 'ldpc'
        N_base: 基准码长
        rates: 码率列表
        snr_range: SNR范围
        num_frames: 每个SNR点的帧数
        max_errors: 提前停止的错误帧数
        test_library: 是否同时测试第三方库
        
    Returns:
        results: 包含所有测试结果的字典
    """
    results = {
        'code_type': code_type,
        'N': N_base,
        'rates': rates,
        'snr_range': snr_range.tolist(),
        'self': {},
        'library': {} if test_library else None
    }
    
    for rate in rates:
        K = int(N_base * rate)
        
        # 测试自实现
        print(f"\n{'#'*70}")
        print(f"Rate = {rate:.3f} (K={K}) - Self Implementation")
        print(f"{'#'*70}")
        
        ber_self, fer_self, stats_self = simulate_snr_curve(
            code_type, N_base, K, snr_range, num_frames, max_errors, use_library=False
        )
        
        results['self'][rate] = {
            'K': K,
            'ber': ber_self,
            'fer': fer_self,
            'stats': stats_self
        }
        
        # 测试第三方库
        if test_library:
            print(f"\n{'#'*70}")
            print(f"Rate = {rate:.3f} (K={K}) - Library Implementation")
            print(f"{'#'*70}")
            
            try:
                ber_lib, fer_lib, stats_lib = simulate_snr_curve(
                    code_type, N_base, K, snr_range, num_frames, max_errors, use_library=True
                )
                
                results['library'][rate] = {
                    'K': K,
                    'ber': ber_lib,
                    'fer': fer_lib,
                    'stats': stats_lib
                }
            except Exception as e:
                print(f"Library test failed: {e}")
                results['library'][rate] = None
    
    return results


def plot_snr_curves(results_polar: Dict, results_ldpc: Dict, output_dir: Path):
    """
    绘制SNR性能曲线
    
    Args:
        results_polar: Polar码测试结果
        results_ldpc: LDPC测试结果
        output_dir: 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    snr_range = np.array(results_polar['snr_range'])
    rates = results_polar['rates']
    
    # 1. BER曲线对比 (自实现)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for rate in rates:
        # Polar
        ber_polar = results_polar['self'][rate]['ber']
        ax.semilogy(snr_range, ber_polar, 'o-', label=f'Polar r={rate:.2f}', linewidth=2)
        
        # LDPC
        ber_ldpc = results_ldpc['self'][rate]['ber']
        ax.semilogy(snr_range, ber_ldpc, 's--', label=f'LDPC r={rate:.2f}', linewidth=2)
    
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title(f'BER vs SNR Comparison (N≈{results_polar["N"]})', fontsize=14)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10, ncol=2)
    ax.set_ylim([1e-6, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ber_vs_snr_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'ber_vs_snr_comparison.png'}")
    plt.close()
    
    # 2. FER曲线对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for rate in rates:
        fer_polar = results_polar['self'][rate]['fer']
        ax.semilogy(snr_range, fer_polar, 'o-', label=f'Polar r={rate:.2f}', linewidth=2)
        
        fer_ldpc = results_ldpc['self'][rate]['fer']
        ax.semilogy(snr_range, fer_ldpc, 's--', label=f'LDPC r={rate:.2f}', linewidth=2)
    
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Frame Error Rate (FER)', fontsize=12)
    ax.set_title(f'FER vs SNR Comparison (N≈{results_polar["N"]})', fontsize=14)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10, ncol=2)
    ax.set_ylim([1e-3, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fer_vs_snr_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fer_vs_snr_comparison.png'}")
    plt.close()
    
    # 3. 分别绘制每个码率的详细对比
    for rate in rates:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # BER
        ber_polar = results_polar['self'][rate]['ber']
        ber_ldpc = results_ldpc['self'][rate]['ber']
        
        ax1.semilogy(snr_range, ber_polar, 'o-', label='Polar (Self)', linewidth=2, markersize=6)
        ax1.semilogy(snr_range, ber_ldpc, 's-', label='LDPC (Self)', linewidth=2, markersize=6)
        
        # 如果有库实现的结果
        if results_polar.get('library') and rate in results_polar['library'] and results_polar['library'][rate]:
            ber_polar_lib = results_polar['library'][rate]['ber']
            ax1.semilogy(snr_range, ber_polar_lib, 'x--', label='Polar (Lib)', linewidth=1.5, markersize=6)
        
        if results_ldpc.get('library') and rate in results_ldpc['library'] and results_ldpc['library'][rate]:
            ber_ldpc_lib = results_ldpc['library'][rate]['ber']
            ax1.semilogy(snr_range, ber_ldpc_lib, '+--', label='LDPC (Lib)', linewidth=1.5, markersize=6)
        
        ax1.set_xlabel('SNR (dB)', fontsize=12)
        ax1.set_ylabel('Bit Error Rate (BER)', fontsize=12)
        ax1.set_title(f'BER vs SNR (Rate={rate:.2f})', fontsize=13)
        ax1.grid(True, which='both', alpha=0.3)
        ax1.legend(fontsize=10)
        
        # FER
        fer_polar = results_polar['self'][rate]['fer']
        fer_ldpc = results_ldpc['self'][rate]['fer']
        
        ax2.semilogy(snr_range, fer_polar, 'o-', label='Polar (Self)', linewidth=2, markersize=6)
        ax2.semilogy(snr_range, fer_ldpc, 's-', label='LDPC (Self)', linewidth=2, markersize=6)
        
        if results_polar.get('library') and rate in results_polar['library'] and results_polar['library'][rate]:
            fer_polar_lib = results_polar['library'][rate]['fer']
            ax2.semilogy(snr_range, fer_polar_lib, 'x--', label='Polar (Lib)', linewidth=1.5, markersize=6)
        
        if results_ldpc.get('library') and rate in results_ldpc['library'] and results_ldpc['library'][rate]:
            fer_ldpc_lib = results_ldpc['library'][rate]['fer']
            ax2.semilogy(snr_range, fer_ldpc_lib, '+--', label='LDPC (Lib)', linewidth=1.5, markersize=6)
        
        ax2.set_xlabel('SNR (dB)', fontsize=12)
        ax2.set_ylabel('Frame Error Rate (FER)', fontsize=12)
        ax2.set_title(f'FER vs SNR (Rate={rate:.2f})', fontsize=13)
        ax2.grid(True, which='both', alpha=0.3)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'snr_curves_rate_{rate:.2f}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / f'snr_curves_rate_{rate:.2f}.png'}")
        plt.close()


def analyze_snr_requirements(results_polar: Dict, results_ldpc: Dict, target_ber: float = 1e-3) -> Dict:
    """
    分析达到目标BER所需的SNR
    
    Args:
        results_polar: Polar码结果
        results_ldpc: LDPC结果
        target_ber: 目标BER阈值
        
    Returns:
        analysis: SNR需求分析字典
    """
    snr_range = np.array(results_polar['snr_range'])
    rates = results_polar['rates']
    
    analysis = {
        'target_ber': target_ber,
        'polar': {},
        'ldpc': {},
        'snr_gap': {}  # Polar相对LDPC需要的额外SNR
    }
    
    print(f"\n{'='*70}")
    print(f"SNR Requirements Analysis (Target BER < {target_ber:.0e})")
    print(f"{'='*70}")
    print(f"{'Rate':<8} {'Polar SNR':<12} {'LDPC SNR':<12} {'Gap (dB)':<12}")
    print(f"{'-'*70}")
    
    for rate in rates:
        ber_polar = np.array(results_polar['self'][rate]['ber'])
        ber_ldpc = np.array(results_ldpc['self'][rate]['ber'])
        
        # 找到达到目标BER的最小SNR
        idx_polar = np.where(ber_polar < target_ber)[0]
        idx_ldpc = np.where(ber_ldpc < target_ber)[0]
        
        snr_polar = snr_range[idx_polar[0]] if len(idx_polar) > 0 else None
        snr_ldpc = snr_range[idx_ldpc[0]] if len(idx_ldpc) > 0 else None
        
        analysis['polar'][rate] = snr_polar
        analysis['ldpc'][rate] = snr_ldpc
        
        if snr_polar is not None and snr_ldpc is not None:
            gap = snr_polar - snr_ldpc
            analysis['snr_gap'][rate] = gap
            print(f"{rate:<8.2f} {snr_polar:<12.1f} {snr_ldpc:<12.1f} {gap:<12.2f}")
        elif snr_polar is None:
            analysis['snr_gap'][rate] = None
            print(f"{rate:<8.2f} {'>'+str(snr_range[-1]):<12} {snr_ldpc if snr_ldpc else '>'+str(snr_range[-1]):<12} {'N/A':<12}")
        else:
            analysis['snr_gap'][rate] = None
            print(f"{rate:<8.2f} {snr_polar:<12.1f} {'>'+str(snr_range[-1]):<12} {'N/A':<12}")
    
    return analysis


def main():
    """主测试函数"""
    
    # 配置参数
    config = {
        'polar_N': 1024,
        'ldpc_N': 1008,  # 需要能被dc=6整除
        'rates': [0.50, 0.67, 0.75, 0.83],  # 低、中、高码率
        'snr_range': np.arange(-2, 6, 1),  # -2 to 5 dB, step=1
        'num_frames': 100,
        'max_errors': 100,
        'test_library': True,
        'output_dir': Path(__file__).parent.parent / 'results' / 'snr_curves'
    }
    
    print(f"\n{'#'*70}")
    print("SNR Performance Curve Testing")
    print(f"{'#'*70}")
    print(f"Configuration:")
    print(f"  Polar N: {config['polar_N']}")
    print(f"  LDPC N:  {config['ldpc_N']}")
    print(f"  Rates:   {config['rates']}")
    print(f"  SNR:     {config['snr_range'][0]:.1f} to {config['snr_range'][-1]:.1f} dB")
    print(f"  Frames:  {config['num_frames']} (max), stop at {config['max_errors']} errors")
    
    # 测试Polar码
    results_polar = test_multiple_rates(
        code_type='polar',
        N_base=config['polar_N'],
        rates=config['rates'],
        snr_range=config['snr_range'],
        num_frames=config['num_frames'],
        max_errors=config['max_errors'],
        test_library=config['test_library']
    )
    
    # 测试LDPC码
    results_ldpc = test_multiple_rates(
        code_type='ldpc',
        N_base=config['ldpc_N'],
        rates=config['rates'],
        snr_range=config['snr_range'],
        num_frames=config['num_frames'],
        max_errors=config['max_errors'],
        test_library=config['test_library']
    )
    
    # 保存原始数据
    output_dir = config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换numpy类型为Python原生类型
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(output_dir / 'polar_results.json', 'w') as f:
        json.dump(convert_to_native(results_polar), f, indent=2)
    print(f"\nSaved: {output_dir / 'polar_results.json'}")
    
    with open(output_dir / 'ldpc_results.json', 'w') as f:
        json.dump(convert_to_native(results_ldpc), f, indent=2)
    print(f"Saved: {output_dir / 'ldpc_results.json'}")
    
    # 绘制曲线
    print(f"\n{'='*70}")
    print("Generating Plots")
    print(f"{'='*70}")
    plot_snr_curves(results_polar, results_ldpc, output_dir)
    
    # 分析SNR需求
    analysis_1e3 = analyze_snr_requirements(results_polar, results_ldpc, target_ber=1e-3)
    analysis_1e5 = analyze_snr_requirements(results_polar, results_ldpc, target_ber=1e-5)
    
    # 保存分析结果
    analysis_combined = {
        'ber_1e-3': analysis_1e3,
        'ber_1e-5': analysis_1e5
    }
    
    with open(output_dir / 'snr_analysis.json', 'w') as f:
        json.dump(convert_to_native(analysis_combined), f, indent=2)
    print(f"\nSaved: {output_dir / 'snr_analysis.json'}")
    
    print(f"\n{'='*70}")
    print("Testing Complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
