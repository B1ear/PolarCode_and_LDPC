"""Inspect polarcodes internal state"""
import numpy as np
from polarcodes import PolarCode, Construct, Decode

N, K = 4, 2

pc = PolarCode(N, K)
Construct(pc, design_SNR=2.0)

# Set LLR
llr = np.array([-10., -10., 10., 10.])
pc.likelihoods = llr

print(f"LLR: {llr}")
print(f"Frozen set: {pc.frozen}")
print(f"Frozen lookup: {pc.frozen_lookup}")
print()

# Decode
Decode(pc, decoder_name='scd')

print(f"After decoding:")
print(f"pc.x: {pc.x}")
print(f"pc.message_received: {pc.message_received}")
print()

# Access decoder's internal state
from polarcodes.SCD import SCD
scd = SCD(pc)

# Manually decode to see matrices
pc2 = PolarCode(N, K)
Construct(pc2, design_SNR=2.0)
pc2.likelihoods = llr

scd2 = SCD(pc2)
result = scd2.decode()

print(f"Decoder result: {result}")
print(f"\nDecoder L matrix:")
for i in range(N):
    print(f"  Position {i}: {scd2.L[i, :]}")

print(f"\nDecoder B matrix:")
for i in range(N):
    print(f"  Position {i}: {scd2.B[i, :]}")
