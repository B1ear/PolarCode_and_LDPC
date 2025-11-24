"""Verify encoding correctness"""
import numpy as np
from src.polar.utils import polar_transform_iterative

# Manual test: N=8, message at positions [3,5,6,7]
N = 8
info_bits = np.array([3, 5, 6, 7])
frozen_bits = np.array([0, 1, 2, 4])

# Message
msg = np.array([1, 0, 1, 1])

# Build u vector
u = np.zeros(N, dtype=int)
u[info_bits] = msg
u[frozen_bits] = 0

print(f"u vector: {u}")
print(f"  Positions 0,1,2,4 (frozen) = {u[[0,1,2,4]]}")
print(f"  Positions 3,5,6,7 (info) = {u[[3,5,6,7]]}")

# Encode using polar transform
x = polar_transform_iterative(u)

print(f"\nCodeword x: {x}")

# Verify: decode x back to u
# Since G_N^2 = I in GF(2), x * G_N = u
u_recovered = polar_transform_iterative(x)

print(f"Recovered u: {u_recovered}")
print(f"Match: {np.array_equal(u, u_recovered)}")

# Now simulate transmission with perfect LLR
# BPSK: bit=0 -> symbol=+1, bit=1 -> symbol=-1
# LLR = 2*symbol/sigma^2, for large LLR: LLR = symbol * large_value
# So: bit=0 -> LLR=+large, bit=1 -> LLR=-large
# Or: LLR = (1 - 2*bit) * large_value

llr = (1 - 2 * x.astype(float)) * 100.0
print(f"\nLLR: {llr}")

# Hard decision on LLR
x_hat = (llr < 0).astype(int)
print(f"Hard decision x_hat: {x_hat}")
print(f"Match with x: {np.array_equal(x, x_hat)}")

# Recover u from x_hat
u_hat = polar_transform_iterative(x_hat)
print(f"\nRecovered u_hat: {u_hat}")
print(f"Match with u: {np.array_equal(u, u_hat)}")
print(f"Message from u_hat: {u_hat[info_bits]}")
print(f"Match with msg: {np.array_equal(msg, u_hat[info_bits])}")
