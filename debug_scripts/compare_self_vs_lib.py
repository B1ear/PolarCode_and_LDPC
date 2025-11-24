"""对比自实现和第三方库LDPC性能"""
import numpy as np
import time
from src.ldpc import LDPCEncoder, BPDecoder
from src.lib_wrappers import LDPCLibWrapper
from src.channel import AWGNChannel

# 初始化
lib = LDPCLibWrapper(504, 252, dv=3, dc=6, seed=42)
k = lib.k
H = lib.get_parity_check_matrix()
G = lib.get_generator_matrix()

enc_self = LDPCEncoder(504, k, H=H, G=G)
dec_self = BPDecoder(H, max_iter=20)

print(f"Comparing Self vs Library (k={k}, max_iter=20)")
print(f"Testing at SNR=-1dB, 20 frames")
print("=" * 60)

# 测试自实现
ch = AWGNChannel(-1.0)
t0 = time.time()
iters = []

for _ in range(20):
    msg = np.random.randint(0, 2, k)
    cw = enc_self.encode(msg)
    llr = ch.transmit(cw, return_llr=True)
    dec, iter_count = dec_self.decode(llr, return_iterations=True)
    iters.append(iter_count)

time_self = time.time() - t0
avg_iters = np.mean(iters)

print(f"\nSelf-Implementation:")
print(f"  Total time: {time_self:.2f}s")
print(f"  Avg time/frame: {time_self/20:.3f}s")
print(f"  Avg iterations: {avg_iters:.1f}")

# 测试第三方库
ch = AWGNChannel(-1.0)
t0 = time.time()

for _ in range(20):
    msg = np.random.randint(0, 2, k)
    cw = lib.encode(msg)
    llr = ch.transmit(cw, return_llr=True)
    dec = lib.decode(llr, max_iter=20)

time_lib = time.time() - t0

print(f"\nThird-Party Library (pyldpc):")
print(f"  Total time: {time_lib:.2f}s")
print(f"  Avg time/frame: {time_lib/20:.3f}s")

print(f"\n{'=' * 60}")
print(f"Comparison:")
print(f"  Self is {time_lib/time_self:.2f}x {'faster' if time_lib > time_self else 'slower'} than library")
print(f"  Speedup breakdown:")
print(f"    Self: {20*k*8/time_self/1e6:.4f} Mbps")
print(f"    Lib:  {20*k*8/time_lib/1e6:.4f} Mbps")
