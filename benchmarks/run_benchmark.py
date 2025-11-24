"""
Main Benchmark Script

运行完整的性能测试对比，包括BER、FER、吞吐量等指标
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
import sys

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polar import PolarEncoder, SCLDecoder
from ldpc import LDPCEncoder, BPDecoder
from channel import AWGNChannel
from utils import plot_ber_curves, save_results
from ber_simulation import run_ber_simulation
from throughput_test import run_throughput_test
from complexity_analysis import analyze_complexity


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_snr_range(snr_str):
    """解析SNR范围字符串: start:stop:step"""
    parts = snr_str.split(':')
    if len(parts) == 3:
        start, stop, step = map(float, parts)
        return np.arange(start, stop + step/2, step)
    else:
        raise ValueError("SNR range format: start:stop:step")


def main():
    parser = argparse.ArgumentParser(description="Run Polar Code and LDPC benchmarks")
    parser.add_argument("--snr-range", type=str, default="-2:6:0.5",
                        help="SNR range in format start:stop:step (default: -2:6:0.5)")
    parser.add_argument("--num-frames", type=int, default=1000,
                        help="Number of frames per SNR point (default: 1000)")
    parser.add_argument("--max-errors", type=int, default=100,
                        help="Stop after this many errors (default: 100)")
    parser.add_argument("--polar-config", type=str, default="config/polar_config.yaml",
                        help="Polar code configuration file")
    parser.add_argument("--ldpc-config", type=str, default="config/ldpc_config.yaml",
                        help="LDPC configuration file")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--skip-ber", action="store_true",
                        help="Skip BER simulation")
    parser.add_argument("--skip-throughput", action="store_true",
                        help="Skip throughput test")
    parser.add_argument("--skip-complexity", action="store_true",
                        help="Skip complexity analysis")
    parser.add_argument("--use-third-party", action="store_true",
                        help="Enable third-party library comparison (pyldpc, py-polar-codes)")
    parser.add_argument("--throughput-iterations", type=int, default=100,
                        help="Number of iterations for throughput test (default: 100)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    
    # 加载配置
    base_path = Path(__file__).parent.parent
    polar_config = load_config(base_path / args.polar_config)
    ldpc_config = load_config(base_path / args.ldpc_config)
    
    # 解析SNR范围
    snr_db_range = parse_snr_range(args.snr_range)
    
    print("=" * 80)
    print("Polar Code and LDPC Performance Benchmark")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  SNR Range: {snr_db_range[0]} to {snr_db_range[-1]} dB (step: {snr_db_range[1] - snr_db_range[0]})")
    print(f"  Frames per SNR: {args.num_frames}")
    print(f"  Max Errors: {args.max_errors}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Third-Party Libraries: {'Enabled' if args.use_third_party else 'Disabled'}")
    
    print(f"\nPolar Code: N={polar_config['encoding']['N']}, K={polar_config['encoding']['K']}")
    print(f"LDPC: n={ldpc_config['encoding']['n']}, k={ldpc_config['encoding']['k']}")
    
    results = {}
    
    # 1. BER仿真
    if not args.skip_ber:
        print("\n" + "=" * 80)
        print("Running BER Simulation...")
        print("=" * 80)
        ber_results = run_ber_simulation(
            snr_db_range=snr_db_range,
            num_frames=args.num_frames,
            max_errors=args.max_errors,
            polar_config=polar_config,
            ldpc_config=ldpc_config,
            output_dir=output_dir,
            use_third_party=args.use_third_party
        )
        results['ber'] = ber_results
    
    # 2. 吞吐量测试
    if not args.skip_throughput:
        print("\n" + "=" * 80)
        print("Running Throughput Test...")
        print("=" * 80)
        throughput_results = run_throughput_test(
            polar_config=polar_config,
            ldpc_config=ldpc_config,
            output_dir=output_dir,
            num_iterations=args.throughput_iterations
        )
        results['throughput'] = throughput_results
    
    # 3. 复杂度分析
    if not args.skip_complexity:
        print("\n" + "=" * 80)
        print("Running Complexity Analysis...")
        print("=" * 80)
        complexity_results = analyze_complexity(
            polar_config=polar_config,
            ldpc_config=ldpc_config,
            output_dir=output_dir
        )
        results['complexity'] = complexity_results
    
    # 保存综合结果
    print("\n" + "=" * 80)
    print("Saving Results...")
    print("=" * 80)
    save_results(results, output_dir / "data" / "benchmark_results.json")
    
    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Figures: {output_dir / 'figures'}")
    print(f"  - Data: {output_dir / 'data'}")


if __name__ == "__main__":
    main()
