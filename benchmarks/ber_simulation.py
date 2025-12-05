"""
BER（误码率）仿真

对Polar码和LDPC码在SNR范围内进行全面的BER/FER仿真。
支持自实现和第三方库实现。
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polar import PolarEncoder, SCDecoder
from ldpc import LDPCEncoder, BPDecoder
from channel import AWGNChannel
from utils.metrics import calculate_ber, calculate_fer
from utils.visualization import plot_ber_curves, save_results


def run_ber_simulation(
    snr_db_range: np.ndarray,
    num_frames: int,
    max_errors: int,
    polar_config: Dict,
    ldpc_config: Dict,
    output_dir: Path,
    use_third_party: bool = False
) -> Dict:
    """
    在SNR范围内运行BER/FER仿真
    
    Args:
        snr_db_range: SNR值数组（dB）
        num_frames: 每个SNR点测试的最大帧数
        max_errors: 达到此帧错误数后停止
        polar_config: Polar码配置字典
        ldpc_config: LDPC配置字典
        output_dir: 结果输出目录
        use_third_party: 如果为True，也测试第三方库实现
        
    Returns:
        包含BER/FER结果的字典
    """
    print(f"\n{'='*60}")
    print("BER/FER Simulation")
    print(f"{'='*60}")
    print(f"SNR Range: {snr_db_range[0]:.1f} to {snr_db_range[-1]:.1f} dB")
    print(f"Frames per SNR: {num_frames} (max), Stop after {max_errors} errors")
    print(f"Use third-party libraries: {use_third_party}")
    
    results = {
        'snr_db': snr_db_range.tolist(),
        'polar': {},
        'ldpc': {}
    }
    
    # 测试Polar码
    print(f"\n{'-'*60}")
    print("Testing Polar Code (Self-Implementation)")
    print(f"{'-'*60}")
    
    polar_ber, polar_fer = simulate_polar(
        snr_db_range, num_frames, max_errors, polar_config
    )
    results['polar']['self'] = {
        'ber': polar_ber.tolist(),
        'fer': polar_fer.tolist()
    }
    
    # 测试LDPC
    print(f"\n{'-'*60}")
    print("Testing LDPC (Self-Implementation)")
    print(f"{'-'*60}")
    
    ldpc_ber, ldpc_fer = simulate_ldpc(
        snr_db_range, num_frames, max_errors, ldpc_config
    )
    results['ldpc']['self'] = {
        'ber': ldpc_ber.tolist(),
        'fer': ldpc_fer.tolist()
    }
    
    # 如果需要，测试第三方库实现
    if use_third_party:
        try:
            from lib_wrappers import PolarLibWrapper, LDPCLibWrapper
            
            print(f"\n{'-'*60}")
            print("Testing Polar Code (Third-Party Library)")
            print(f"{'-'*60}")
            
            polar_lib_ber, polar_lib_fer = simulate_polar_lib(
                snr_db_range, num_frames, max_errors, polar_config
            )
            results['polar']['library'] = {
                'ber': polar_lib_ber.tolist(),
                'fer': polar_lib_fer.tolist()
            }
            
            print(f"\n{'-'*60}")
            print("Testing LDPC (Third-Party Library)")
            print(f"{'-'*60}")
            
            ldpc_lib_ber, ldpc_lib_fer = simulate_ldpc_lib(
                snr_db_range, num_frames, max_errors, ldpc_config
            )
            results['ldpc']['library'] = {
                'ber': ldpc_lib_ber.tolist(),
                'fer': ldpc_lib_fer.tolist()
            }
            
        except ImportError as e:
            print(f"\nWarning: Third-party libraries not available: {e}")
    
    # Plot results
    print(f"\n{'-'*60}")
    print("Generating Plots")
    print(f"{'-'*60}")
    
    plot_ber_results(results, output_dir)
    
    # Save results
    save_results(results, output_dir / "data" / "ber_simulation_results.json")
    
    return results


def simulate_polar(
    snr_db_range: np.ndarray,
    num_frames: int,
    max_errors: int,
    config: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate Polar code BER/FER"""
    
    N = config['encoding']['N']
    K = config['encoding']['K']
    
    print(f"Polar: N={N}, K={K}, rate={K/N:.3f}")
    
    # Use polarcodes library frozen bits for fair comparison
    from lib_wrappers import PolarLibWrapper
    lib = PolarLibWrapper(N, K, 2.0)
    frozen_bits = lib.get_frozen_bits_positions()
    
    # Create encoder and decoder
    encoder = PolarEncoder(N, K, frozen_bits=frozen_bits)
    decoder = SCDecoder(N, K, frozen_bits=frozen_bits)
    
    ber_list = []
    fer_list = []
    
    for snr_db in snr_db_range:
        channel = AWGNChannel(snr_db=snr_db, seed=None)
        
        total_bits = 0
        error_bits = 0
        frame_errors = 0
        frames_tested = 0
        
        start_time = time.time()
        
        for _ in range(num_frames):
            # Generate random message
            message = np.random.randint(0, 2, K)
            
            # Encode
            codeword = encoder.encode(message)
            
            # Transmit through channel
            llr = channel.transmit(codeword, return_llr=True)
            
            # Decode
            decoded = decoder.decode(llr)
            
            # Calculate errors
            bit_errors = np.sum(message != decoded)
            total_bits += K
            error_bits += bit_errors
            
            if bit_errors > 0:
                frame_errors += 1
            
            frames_tested += 1
            
            # Early stopping
            if frame_errors >= max_errors:
                break
        
        elapsed = time.time() - start_time
        
        ber = error_bits / total_bits if total_bits > 0 else 0
        fer = frame_errors / frames_tested if frames_tested > 0 else 0
        
        ber_list.append(ber)
        fer_list.append(fer)
        
        print(f"  SNR={snr_db:4.1f}dB: BER={ber:.6f}, FER={fer:.4f}, "
              f"Frames={frames_tested}, Time={elapsed:.2f}s")
    
    return np.array(ber_list), np.array(fer_list)


def simulate_ldpc(
    snr_db_range: np.ndarray,
    num_frames: int,
    max_errors: int,
    config: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate LDPC BER/FER"""
    
    n = config['encoding']['n']
    k = config['encoding']['k']
    dv = config['construction'].get('dv', 3)
    dc = config['construction'].get('dc', 6)
    max_iter = config['decoding'].get('max_iterations', 50)
    
    print(f"LDPC: n={n}, k={k}, rate={k/n:.3f}, dv={dv}, dc={dc}")
    
    # Use pyldpc H and G for fair comparison with library
    from lib_wrappers import LDPCLibWrapper
    lib = LDPCLibWrapper(n, k, dv=dv, dc=dc, seed=42)
    k_actual = lib.k
    H = lib.get_parity_check_matrix()
    G = lib.get_generator_matrix()
    
    if k_actual != k:
        print(f"  Note: Using actual k={k_actual} from pyldpc (requested {k})")
    
    # Use actual k from library (pyldpc may adjust k slightly)
    k = k_actual
    
    # Create encoder with pyldpc's H and G matrices, test our implementation
    encoder = LDPCEncoder(n, k, H=H, G=G)
    decoder = BPDecoder(H, max_iter=max_iter)
    
    ber_list = []
    fer_list = []
    
    for snr_db in snr_db_range:
        channel = AWGNChannel(snr_db=snr_db, seed=None)
        
        total_bits = 0
        error_bits = 0
        frame_errors = 0
        frames_tested = 0
        
        start_time = time.time()
        
        for _ in range(num_frames):
            # Generate random message
            message = np.random.randint(0, 2, k)
            
            # Encode (using self-implementation with pyldpc's G matrix)
            codeword = encoder.encode(message)
            
            # Transmit through channel
            llr = channel.transmit(codeword, return_llr=True)
            
            # Decode (returns full codeword)
            decoded_cw = decoder.decode(llr)
            decoded_msg = decoded_cw[:k]  # Extract message bits
            
            # Calculate errors
            bit_errors = np.sum(message != decoded_msg)
            total_bits += k
            error_bits += bit_errors
            
            if bit_errors > 0:
                frame_errors += 1
            
            frames_tested += 1
            
            # Early stopping
            if frame_errors >= max_errors:
                break
        
        elapsed = time.time() - start_time
        
        ber = error_bits / total_bits if total_bits > 0 else 0
        fer = frame_errors / frames_tested if frames_tested > 0 else 0
        
        ber_list.append(ber)
        fer_list.append(fer)
        
        print(f"  SNR={snr_db:4.1f}dB: BER={ber:.6f}, FER={fer:.4f}, "
              f"Frames={frames_tested}, Time={elapsed:.2f}s")
    
    return np.array(ber_list), np.array(fer_list)


def simulate_polar_lib(
    snr_db_range: np.ndarray,
    num_frames: int,
    max_errors: int,
    config: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate Polar code using third-party library"""
    
    from lib_wrappers import PolarLibWrapper
    
    N = config['encoding']['N']
    K = config['encoding']['K']
    design_snr = config['construction'].get('design_snr_db', 2.0)
    
    print(f"Polar (Library): N={N}, K={K}, design_SNR={design_snr}dB")
    
    wrapper = PolarLibWrapper(N, K, design_snr_db=design_snr)
    
    ber_list = []
    fer_list = []
    
    for snr_db in snr_db_range:
        channel = AWGNChannel(snr_db=snr_db, seed=None)
        
        total_bits = 0
        error_bits = 0
        frame_errors = 0
        frames_tested = 0
        
        start_time = time.time()
        
        for _ in range(num_frames):
            message = np.random.randint(0, 2, K)
            codeword = wrapper.encode(message)
            llr = channel.transmit(codeword, return_llr=True)
            decoded = wrapper.decode(llr)
            
            bit_errors = np.sum(message != decoded)
            total_bits += K
            error_bits += bit_errors
            
            if bit_errors > 0:
                frame_errors += 1
            
            frames_tested += 1
            
            if frame_errors >= max_errors:
                break
        
        elapsed = time.time() - start_time
        
        ber = error_bits / total_bits if total_bits > 0 else 0
        fer = frame_errors / frames_tested if frames_tested > 0 else 0
        
        ber_list.append(ber)
        fer_list.append(fer)
        
        print(f"  SNR={snr_db:4.1f}dB: BER={ber:.6f}, FER={fer:.4f}, "
              f"Frames={frames_tested}, Time={elapsed:.2f}s")
    
    return np.array(ber_list), np.array(fer_list)


def simulate_ldpc_lib(
    snr_db_range: np.ndarray,
    num_frames: int,
    max_errors: int,
    config: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate LDPC using third-party library"""
    
    from lib_wrappers import LDPCLibWrapper
    
    n = config['encoding']['n']
    k = config['encoding']['k']
    dv = config['construction'].get('dv', 3)
    dc = config['construction'].get('dc', 6)
    max_iter = config['decoding'].get('max_iterations', 20)
    
    print(f"LDPC (Library): n={n}, k={k}, dv={dv}, dc={dc}, max_iter={max_iter}")
    
    wrapper = LDPCLibWrapper(n, k, dv=dv, dc=dc, seed=42)
    k_actual = wrapper.k
    
    if k_actual != k:
        print(f"  Note: Actual k={k_actual} (requested {k})")
    
    ber_list = []
    fer_list = []
    
    for snr_db in snr_db_range:
        channel = AWGNChannel(snr_db=snr_db, seed=None)
        
        total_bits = 0
        error_bits = 0
        frame_errors = 0
        frames_tested = 0
        
        start_time = time.time()
        
        for _ in range(num_frames):
            message = np.random.randint(0, 2, k_actual)
            codeword = wrapper.encode(message)
            llr = channel.transmit(codeword, return_llr=True)
            decoded = wrapper.decode(llr, max_iter=max_iter)
            
            bit_errors = np.sum(message != decoded)
            total_bits += k_actual
            error_bits += bit_errors
            
            if bit_errors > 0:
                frame_errors += 1
            
            frames_tested += 1
            
            if frame_errors >= max_errors:
                break
        
        elapsed = time.time() - start_time
        
        ber = error_bits / total_bits if total_bits > 0 else 0
        fer = frame_errors / frames_tested if frames_tested > 0 else 0
        
        ber_list.append(ber)
        fer_list.append(fer)
        
        print(f"  SNR={snr_db:4.1f}dB: BER={ber:.6f}, FER={fer:.4f}, "
              f"Frames={frames_tested}, Time={elapsed:.2f}s")
    
    return np.array(ber_list), np.array(fer_list)


def plot_ber_results(results: Dict, output_dir: Path):
    """Plot BER and FER curves"""
    
    import matplotlib.pyplot as plt
    
    snr_db = np.array(results['snr_db'])
    
    # Plot BER
    plt.figure(figsize=(10, 6))
    
    # Polar
    if 'self' in results['polar']:
        ber = np.array(results['polar']['self']['ber'])
        # Replace zeros with small value for log plot
        ber = np.where(ber == 0, 1e-6, ber)
        plt.semilogy(snr_db, ber, 'o-', label='Polar (Self)', linewidth=2)
    
    if 'library' in results['polar']:
        ber = np.array(results['polar']['library']['ber'])
        # Replace zeros with small value for log plot
        ber = np.where(ber == 0, 1e-6, ber)
        plt.semilogy(snr_db, ber, 's--', label='Polar (Library)', linewidth=2, markersize=8)
    
    # LDPC
    if 'self' in results['ldpc']:
        ber = np.array(results['ldpc']['self']['ber'])
        # Replace zeros with small value for log plot
        ber = np.where(ber == 0, 1e-6, ber)
        plt.semilogy(snr_db, ber, '^-', label='LDPC (Self)', linewidth=2)
    
    if 'library' in results['ldpc']:
        ber = np.array(results['ldpc']['library']['ber'])
        # Replace zeros with small value for log plot
        ber = np.where(ber == 0, 1e-6, ber)
        plt.semilogy(snr_db, ber, 'd--', label='LDPC (Library)', linewidth=2, markersize=8)
    
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('BER Performance Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    # Add note about zero values
    plt.text(0.02, 0.02, 'Note: Zero BER shown as 1e-6', 
             transform=plt.gca().transAxes, fontsize=8, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    plt.tight_layout()
    
    ber_path = output_dir / "figures" / "ber_curves.png"
    plt.savefig(ber_path, dpi=300, bbox_inches='tight')
    print(f"  Saved BER plot: {ber_path}")
    plt.close()
    
    # Plot FER
    plt.figure(figsize=(10, 6))
    
    if 'self' in results['polar']:
        fer = np.array(results['polar']['self']['fer'])
        # Replace zeros with small value for log plot
        fer = np.where(fer == 0, 1e-4, fer)
        plt.semilogy(snr_db, fer, 'o-', label='Polar (Self)', linewidth=2)
    
    if 'library' in results['polar']:
        fer = np.array(results['polar']['library']['fer'])
        # Replace zeros with small value for log plot
        fer = np.where(fer == 0, 1e-4, fer)
        plt.semilogy(snr_db, fer, 's--', label='Polar (Library)', linewidth=2, markersize=8)
    
    if 'self' in results['ldpc']:
        fer = np.array(results['ldpc']['self']['fer'])
        # Replace zeros with small value for log plot
        fer = np.where(fer == 0, 1e-4, fer)
        plt.semilogy(snr_db, fer, '^-', label='LDPC (Self)', linewidth=2)
    
    if 'library' in results['ldpc']:
        fer = np.array(results['ldpc']['library']['fer'])
        # Replace zeros with small value for log plot
        fer = np.where(fer == 0, 1e-4, fer)
        plt.semilogy(snr_db, fer, 'd--', label='LDPC (Library)', linewidth=2, markersize=8)
    
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Frame Error Rate (FER)', fontsize=12)
    plt.title('FER Performance Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    # Add note about zero values
    plt.text(0.02, 0.02, 'Note: Zero FER shown as 1e-4', 
             transform=plt.gca().transAxes, fontsize=8, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    plt.tight_layout()
    
    fer_path = output_dir / "figures" / "fer_curves.png"
    plt.savefig(fer_path, dpi=300, bbox_inches='tight')
    print(f"  Saved FER plot: {fer_path}")
    plt.close()


if __name__ == "__main__":
    # Test simulation
    print("Testing BER Simulation Module...")
    
    snr_range = np.arange(0, 5, 1.0)
    
    polar_config = {
        'encoding': {'N': 128, 'K': 64},
        'construction': {'design_snr_db': 2.0}
    }
    
    ldpc_config = {
        'encoding': {'n': 120, 'k': 60, 'dv': 3, 'dc': 6},
        'decoding': {'max_iterations': 50}
    }
    
    # Use absolute path relative to this file
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    
    results = run_ber_simulation(
        snr_db_range=snr_range,
        num_frames=20,
        max_errors=10,
        polar_config=polar_config,
        ldpc_config=ldpc_config,
        output_dir=output_dir,
        use_third_party=True  # Enable third-party library comparison
    )
    
    print("\n✓ BER Simulation test passed!")
