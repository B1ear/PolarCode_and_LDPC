"""Compare library and self decoder with same LLR input"""
import numpy as np
from src.polar.decoder import SCDecoder
from src.lib_wrappers.polar_wrapper import PolarLibWrapper

N, K = 8, 4

lib = PolarLibWrapper(N, K, 2.0)
frozen_bits = lib.get_frozen_bits_positions()

dec_self = SCDecoder(N, K, frozen_bits=frozen_bits)

# Test with a specific LLR vector (not from encoding)
# Use asymmetric values to avoid cancellation
test_llrs = [
    np.array([10.0, -5.0, 8.0, -12.0, 6.0, -9.0, 7.0, -11.0]),
    np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]),
    np.array([-100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0]),
    np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
]

print(f"N={N}, K={K}")
print(f"Frozen bits: {sorted(frozen_bits)}")
print(f"Info bits: {sorted(lib.get_info_bits_positions())}")
print()

for i, llr in enumerate(test_llrs):
    print(f"Test {i+1}:")
    print(f"  LLR: {llr}")
    
    dec_self_out = dec_self.decode(llr)
    dec_lib_out = lib.decode(llr)
    
    # Also get full u vectors
    u_self = dec_self._decode_recursive(llr, dec_self.frozen_mask)
    
    print(f"  Full u (self): {u_self}")
    print(f"  Decoded (self): {dec_self_out}")
    print(f"  Decoded (lib):  {dec_lib_out}")
    print(f"  Match: {np.array_equal(dec_self_out, dec_lib_out)}")
    print()
