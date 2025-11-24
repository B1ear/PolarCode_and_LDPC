"""
Analyze LDPC Decoding Performance
诊断BP解码器为何比Polar慢93倍
"""
import numpy as np
import time
from src.ldpc import LDPCEncoder, BPDecoder
from src.channel import AWGNChannel
from src.lib_wrappers import LDPCLibWrapper

# 配置
n, k = 504, 252
dv, dc = 3, 6
snr_db = 3.0
num_frames = 50

print("="*70)
print("LDPC Performance Analysis")
print("="*70)

# 初始化
lib = LDPCLibWrapper(n, k, dv=dv, dc=dc, seed=42)
k_actual = lib.k
H = lib.get_parity_check_matrix()
G = lib.get_generator_matrix()

print(f"\nConfiguration:")
print(f"  n={n}, k_actual={k_actual}, rate={k_actual/n:.3f}")
print(f"  H.shape={H.shape}, dv={dv}, dc={dc}")
print(f"  SNR={snr_db}dB, frames={num_frames}")

# 创建encoder和decoder
encoder = LDPCEncoder(n, k_actual, H=H, G=G)
decoder = BPDecoder(H, max_iter=50, early_stop=True)

channel = AWGNChannel(snr_db)

# 统计数据
iteration_counts = []
decode_times = []
encode_times = []

print(f"\n{'='*70}")
print("Running test...")
print(f"{'='*70}\n")

for i in range(num_frames):
    # 生成消息
    msg = np.random.randint(0, 2, k_actual)
    
    # 编码
    t0 = time.time()
    codeword = encoder.encode(msg)
    encode_time = time.time() - t0
    encode_times.append(encode_time)
    
    # 传输
    llr = channel.transmit(codeword, return_llr=True)
    
    # 解码（返回迭代次数）
    t0 = time.time()
    decoded, iters = decoder.decode(llr, return_iterations=True)
    decode_time = time.time() - t0
    
    decode_times.append(decode_time)
    iteration_counts.append(iters)
    
    # 验证
    errors = np.sum(decoded[:k_actual] != msg)
    
    if i < 5:  # 打印前5帧详情
        print(f"Frame {i+1:2d}: iters={iters:2d}, decode_time={decode_time*1000:.2f}ms, "
              f"encode_time={encode_time*1000:.2f}ms, errors={errors}")

# 分析结果
print(f"\n{'='*70}")
print("Analysis Results")
print(f"{'='*70}\n")

iteration_counts = np.array(iteration_counts)
decode_times = np.array(decode_times)
encode_times = np.array(encode_times)

print(f"Iteration Statistics:")
print(f"  Mean: {iteration_counts.mean():.2f}")
print(f"  Median: {np.median(iteration_counts):.0f}")
print(f"  Min: {iteration_counts.min()}")
print(f"  Max: {iteration_counts.max()}")
print(f"  Std: {iteration_counts.std():.2f}")
print(f"  Early stop rate: {(iteration_counts < 50).sum()}/{num_frames} ({(iteration_counts < 50).mean()*100:.1f}%)")

print(f"\nDecode Time Statistics:")
print(f"  Mean: {decode_times.mean()*1000:.2f}ms")
print(f"  Median: {np.median(decode_times)*1000:.2f}ms")
print(f"  Min: {decode_times.min()*1000:.2f}ms")
print(f"  Max: {decode_times.max()*1000:.2f}ms")
print(f"  Total: {decode_times.sum():.2f}s")

print(f"\nEncode Time Statistics:")
print(f"  Mean: {encode_times.mean()*1000:.2f}ms")
print(f"  Median: {np.median(encode_times)*1000:.2f}ms")
print(f"  Total: {encode_times.sum():.2f}s")

# 计算吞吐量
total_bits = num_frames * k_actual
decode_throughput = total_bits / decode_times.sum() / 1e6  # Mbps
encode_throughput = total_bits / encode_times.sum() / 1e6  # Mbps

print(f"\nThroughput:")
print(f"  Encoding: {encode_throughput:.4f} Mbps")
print(f"  Decoding: {decode_throughput:.4f} Mbps")

# 分析瓶颈
print(f"\n{'='*70}")
print("Bottleneck Analysis")
print(f"{'='*70}\n")

# 理论复杂度
ops_per_iter = n * dv + H.shape[0] * dc  # 变量节点更新 + 校验节点更新
total_ops = ops_per_iter * iteration_counts.mean()
ops_per_sec = total_ops * num_frames / decode_times.sum()

print(f"Theoretical Complexity:")
print(f"  Ops per iteration: {ops_per_iter:,}")
print(f"  Average total ops: {total_ops:,.0f}")
print(f"  Ops/second: {ops_per_sec/1e6:.2f} MOps/s")

print(f"\nTime per iteration:")
print(f"  {decode_times.mean()/iteration_counts.mean()*1000:.3f} ms/iter")

# 关键发现
print(f"\n{'='*70}")
print("Key Findings")
print(f"{'='*70}\n")

if iteration_counts.mean() < 50:
    print(f"✓ Early stopping is working (avg {iteration_counts.mean():.1f}/{50} iterations)")
else:
    print(f"✗ Early stopping NOT working effectively (hitting max_iter)")

if decode_times.mean() > 0.5:  # 超过500ms
    print(f"✗ Decode time is very slow ({decode_times.mean()*1000:.0f}ms/frame)")
    print(f"  Possible causes:")
    print(f"  - Python loops (not vectorized)")
    print(f"  - Inefficient tanh/arctanh calculations")
    print(f"  - Message passing overhead")
else:
    print(f"✓ Decode time is acceptable ({decode_times.mean()*1000:.0f}ms/frame)")

# 对比Polar
print(f"\nComparison with Polar:")
polar_decode_mbps = 0.027  # 从benchmark结果
ldpc_decode_mbps = decode_throughput
slowdown = polar_decode_mbps / ldpc_decode_mbps
print(f"  Polar: {polar_decode_mbps:.4f} Mbps")
print(f"  LDPC:  {ldpc_decode_mbps:.4f} Mbps")
print(f"  Slowdown: {slowdown:.1f}x")
