"""Compare step by step with polarcodes"""
import numpy as np
from src.polar.encoder import PolarEncoder
from src.polar.decoder import SCDecoder
from src.lib_wrappers.polar_wrapper import PolarLibWrapper

N, K = 4, 2

lib = PolarLibWrapper(N, K, 2.0)
frozen_bits = lib.get_frozen_bits_positions()

enc = PolarEncoder(N, K, frozen_bits=frozen_bits)
dec_self = SCDecoder(N, K, frozen_bits=frozen_bits)

# Test one specific message
msg = np.array([1, 0])
cw = enc.encode(msg)
llr = (1 - 2 * cw.astype(float)) * 10.0  # Use smaller LLR

print(f"Message: {msg}")
print(f"Codeword: {cw}")
print(f"LLR: {llr}")
print(f"Frozen bits: {sorted(frozen_bits)}")
print(f"Info bits: {sorted(lib.get_info_bits_positions())}")
print()

# Decode with library
dec_lib = lib.decode(llr)
print(f"Library decoded: {dec_lib}")

# Decode with self and inspect
dec_self_result = dec_self.decode(llr)
print(f"Self decoded: {dec_self_result}")
print()

# Check internal state
print("Self decoder internal state:")
print(f"L matrix:")
for i in range(N):
    print(f"  Position {i}: {dec_self.L[i, :]}")
print()
print(f"B matrix:")
for i in range(N):
    print(f"  Position {i}: {dec_self.B[i, :]}")
print()
print(f"Final u: {dec_self.B[:, dec_self.n]}")
print(f"Info positions {dec_self.info_bits}: {dec_self.B[:, dec_self.n][dec_self.info_bits]}")

# Test with polarcodes directly
print("\n=== Testing polarcodes directly ===")
from polarcodes import PolarCode, Construct, Encode, Decode

pc = PolarCode(N, K)
Construct(pc, design_SNR=2.0)
pc.likelihoods = llr
Decode(pc, decoder_name='scd')

print(f"Polarcodes decoded u: {pc.x}")
print(f"Polarcodes message: {pc.message_received}")
