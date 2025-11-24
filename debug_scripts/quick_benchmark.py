"""
Quick Benchmark Script

快速运行benchmark，使用较少的迭代次数，适合测试和演示。
"""

import subprocess
import sys

# 快速benchmark配置
cmd = [
    sys.executable,
    "benchmarks/run_benchmark.py",
    "--snr-range", "1:5:1",           # 5个SNR点
    "--num-frames", "50",              # 每个SNR点50帧
    "--max-errors", "20",              # 20个错误后停止
    "--throughput-iterations", "50",   # 吞吐量测试50次迭代
    "--use-third-party"                # 启用第三方库对比
]

print("=" * 70)
print("Quick Benchmark Mode")
print("=" * 70)
print("\n配置:")
print("  SNR范围: 1-5 dB (步长1)")
print("  BER测试: 50帧/SNR点, 最多20错误")
print("  吞吐量测试: 50次迭代")
print("  第三方库: 启用")
print("\n预计耗时: ~3-5分钟")
print("=" * 70)
print()

# 运行benchmark
subprocess.run(cmd)
