"""Trace LLR updates in detail"""
import numpy as np
from src.polar.utils import bit_reverse

# Manual simulation of N=4, K=2
N = 4
n = 2
frozen_set = {0, 2}

# LLR from codeword [1,1,0,0]
llr_input = np.array([-100., -100., 100., 100.])

# Initialize matrices
L = np.zeros((N, n + 1), dtype=np.float64)
B = np.zeros((N, n + 1), dtype=np.int8)

L[:, 0] = llr_input

print(f"Initial L[:, 0]: {L[:, 0]}")
print()

# Process in bit-reversed order
for i in range(N):
    bit_idx = bit_reverse(i, n)
    print(f"=== Iteration {i}: bit_idx={bit_idx} ===")
    
    # Determine active_llr_level
    if bit_idx == 0:
        active_level = 0
    else:
        level = 0
        temp = bit_idx
        while temp & 1 == 0:
            level += 1
            temp >>= 1
        active_level = n - level
    
    print(f"Active LLR level: {active_level}")
    
    # Update LLRs
    for depth in range(active_level, n):
        block_size = 2 ** (depth + 1)
        half_block = block_size // 2
        block_start = (bit_idx // block_size) * block_size
        
        print(f"  Depth {depth}: block_size={block_size}, half_block={half_block}, block_start={block_start}")
        
        if (bit_idx % block_size) < half_block:
            # Upper branch
            top_idx = block_start + (bit_idx % half_block)
            btm_idx = top_idx + half_block
            
            top_llr = L[top_idx, depth]
            btm_llr = L[btm_idx, depth]
            
            result = np.sign(top_llr) * np.sign(btm_llr) * min(abs(top_llr), abs(btm_llr))
            L[bit_idx, depth + 1] = result
            
            print(f"    UPPER: top_idx={top_idx}, btm_idx={btm_idx}")
            print(f"    f({top_llr:.1f}, {btm_llr:.1f}) = {result:.1f}")
        else:
            # Lower branch
            btm_idx = block_start + (bit_idx % half_block)
            top_idx = btm_idx - half_block
            
            top_llr = L[top_idx, depth]
            btm_llr = L[bit_idx, depth]
            top_bit = B[top_idx, depth + 1]
            
            result = btm_llr + (1 - 2 * top_bit) * top_llr
            L[bit_idx, depth + 1] = result
            
            print(f"    LOWER: top_idx={top_idx}, btm_idx={btm_idx}, top_bit={top_bit}")
            print(f"    g({btm_llr:.1f}, {top_llr:.1f}, {top_bit}) = {result:.1f}")
    
    # Hard decision
    if bit_idx in frozen_set:
        B[bit_idx, n] = 0
        print(f"  Decision: 0 (frozen)")
    else:
        B[bit_idx, n] = 0 if L[bit_idx, n] >= 0 else 1
        print(f"  Decision: {B[bit_idx, n]} (LLR={L[bit_idx, n]:.1f})")
    
    print()

print(f"Final B[:, n]: {B[:, n]}")
print(f"Info bits [1, 3]: {B[[1,3], n]}")
