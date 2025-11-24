"""Check if bit-reversal is needed"""
import numpy as np

def bit_reverse_indices(n):
    """Generate bit-reversed indices for length 2^n"""
    N = 2 ** n
    indices = np.arange(N)
    reversed_indices = np.zeros(N, dtype=int)
    
    for i in range(N):
        # Reverse the bits
        rev = 0
        for j in range(n):
            if i & (1 << j):
                rev |= (1 << (n - 1 - j))
        reversed_indices[i] = rev
    
    return reversed_indices

# Check for N=8
n = 3  # N = 2^3 = 8
N = 8

br_indices = bit_reverse_indices(n)
print(f"N={N}, n={n}")
print(f"Bit-reverse indices: {br_indices}")
print(f"  0 -> {br_indices[0]}")
print(f"  1 -> {br_indices[1]}")
print(f"  2 -> {br_indices[2]}")
print(f"  3 -> {br_indices[3]}")
print(f"  4 -> {br_indices[4]}")
print(f"  5 -> {br_indices[5]}")
print(f"  6 -> {br_indices[6]}")
print(f"  7 -> {br_indices[7]}")

# Test with our encoder/decoder
from src.polar.encoder import PolarEncoder
from src.lib_wrappers.polar_wrapper import PolarLibWrapper

lib = PolarLibWrapper(N, 4, 2.0)
enc = PolarEncoder(N, 4, frozen_bits=lib.get_frozen_bits_positions())

msg = np.array([1, 0, 1, 1])
cw_self = enc.encode(msg)
cw_lib = lib.encode(msg)

print(f"\nCodewords match: {np.array_equal(cw_self, cw_lib)}")

# Try applying bit-reversal to LLR input to decoder?
from src.polar.decoder import SCDecoder
dec = SCDecoder(N, 4, frozen_bits=lib.get_frozen_bits_positions())

llr = (1 - 2 * cw_self.astype(float)) * 1000.0

# Try reversing LLR order?
llr_reversed = llr[br_indices]

print(f"\nOriginal LLR: {llr}")
print(f"Reversed LLR: {llr_reversed}")

dec_normal = dec.decode(llr)
dec_reversed = dec.decode(llr_reversed)
dec_lib = lib.decode(llr)

print(f"\nDecoded (normal): {dec_normal}")
print(f"Decoded (reversed): {dec_reversed}")
print(f"Decoded (lib): {dec_lib}")
print(f"Expected: {msg}")

print(f"\nMatch (normal): {np.array_equal(msg, dec_normal)}")
print(f"Match (reversed): {np.array_equal(msg, dec_reversed)}")
print(f"Match (lib): {np.array_equal(msg, dec_lib)}")
