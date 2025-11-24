"""
Polar Code Decoders

SC (Successive Cancellation) decoder using standard iterative bit-reversed algorithm.
SCL (Successive Cancellation List) decoder is a simple wrapper for now.
"""

import numpy as np
from typing import Optional

from .utils import bit_reverse
class SCDecoder:
    """Standard iterative SC decoder - direct translation from polarcodes library.
    
    This implementation is a faithful translation of the polarcodes SCD class,
    following the exact same logic and structure.
    """

    def __init__(self, N: int, K: int, frozen_bits: Optional[np.ndarray] = None):
        assert N > 0 and (N & (N - 1)) == 0, "N must be a power of 2"
        assert 0 < K < N, "K must be in (0, N)"

        self.N = N
        self.K = K
        self.n = int(np.log2(N))  # Tree depth

        if frozen_bits is None:
            from .utils import generate_frozen_bits
            self.frozen_bits, self.info_bits = generate_frozen_bits(N, K)
        else:
            self.frozen_bits = np.array(frozen_bits, dtype=int)
            self.info_bits = np.setdiff1d(np.arange(N), self.frozen_bits)
        
        # Convert frozen_bits to set for O(1) lookup
        self.frozen_set = set(self.frozen_bits)
        
        # LLR and bit matrices - initialize with NaN like polarcodes
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan, dtype=np.float64)

    def decode(self, llr_input: np.ndarray) -> np.ndarray:
        """SC decode - exact translation from polarcodes.
        
        Args:
            llr_input: Channel LLR values, length N (positive means bit=0 more likely)
        
        Returns:
            Decoded information bits, length K
        """
        llr_input = np.asarray(llr_input, dtype=np.float64)
        assert llr_input.shape == (self.N,), f"expected LLR shape ({self.N},), got {llr_input.shape}"

        # Initialize LLR matrix with channel observations
        self.L[:, 0] = llr_input
        
        # Decode bits in bit-reversed order (same as polarcodes)
        for i in range(self.N):
            l = bit_reverse(i, self.n)
            
            # Update LLRs
            self._update_llrs(l)
            
            # Make hard decision
            if l in self.frozen_set:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = self._hard_decision(self.L[l, self.n])
            
            # Propagate bit decision
            self._update_bits(l)
        
        # Extract information bits
        u_decoded = self.B[:, self.n].astype(int)
        return u_decoded[self.info_bits]

    def _update_llrs(self, l: int):
        """Update LLRs - exact translation from polarcodes.
        
        Args:
            l: Bit index to update LLRs for
        """
        # Iterate over relevant depths
        for s in range(self.n - self._active_llr_level(l, self.n), self.n):
            block_size = int(2 ** (s + 1))
            branch_size = int(block_size / 2)
            
            # Iterate over all positions at this depth
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:  # upper branch
                    top_llr = self.L[j, s]
                    btm_llr = self.L[j + branch_size, s]
                    self.L[j, s + 1] = self._upper_llr(top_llr, btm_llr)
                else:  # lower branch
                    btm_llr = self.L[j, s]
                    top_llr = self.L[j - branch_size, s]
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = self._lower_llr(btm_llr, top_llr, top_bit)
    
    def _update_bits(self, l: int):
        """Update bits - exact translation from polarcodes.
        
        Args:
            l: Bit index to update bits for
        """
        # Early return for upper half
        if l < self.N / 2:
            return
        
        # Iterate over depths from n down to active level
        for s in range(self.n, self.n - self._active_bit_level(l, self.n), -1):
            block_size = int(2 ** s)
            branch_size = int(block_size / 2)
            
            # Iterate backwards from l
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:  # lower branch
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    self.B[j, s - 1] = self.B[j, s]
    
    def _hard_decision(self, y: float) -> int:
        """Hard decision on LLR."""
        return 0 if y >= 0 else 1
    
    def _upper_llr(self, l1: float, l2: float) -> float:
        """Upper branch LLR update (f function).
        
        Simplified version without log-domain operations.
        Uses min-sum approximation: f(a,b) = sign(a)*sign(b)*min(|a|,|b|)
        """
        return np.sign(l1) * np.sign(l2) * min(abs(l1), abs(l2))
    
    def _lower_llr(self, l1: float, l2: float, b: float) -> float:
        """Lower branch LLR update (g function).
        
        Exact match to polarcodes implementation.
        
        Args:
            l1: bottom LLR (btm_llr in polarcodes)
            l2: top LLR (top_llr in polarcodes)
            b: top bit decision
        
        Returns:
            Updated LLR
        """
        if b == 0:
            return l1 + l2  # btm + top
        else:  # b == 1
            return l1 - l2  # btm - top
    
    def _active_llr_level(self, i: int, n: int) -> int:
        """Find the first 1 in binary expansion of i.
        
        Exact translation from polarcodes.
        """
        mask = 2 ** (n - 1)
        count = 1
        for k in range(n):
            if (mask & i) == 0:
                count += 1
                mask >>= 1
            else:
                break
        return min(count, n)
    
    def _active_bit_level(self, i: int, n: int) -> int:
        """Find the first 0 in binary expansion of i.
        
        Exact translation from polarcodes.
        """
        mask = 2 ** (n - 1)
        count = 1
        for k in range(n):
            if (mask & i) > 0:
                count += 1
                mask >>= 1
            else:
                break
        return min(count, n)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"SCDecoder(N={self.N}, K={self.K})"


class SCLDecoder:
    """Simple SCL decoder wrapper.

    当前实现：直接调用 SC 解码（等价于 L=1 的 SCL）。
    这样可以先保证接口正确、结果可靠，再在此基础上实现真正的列表解码。
    """

    def __init__(self, N: int, K: int, list_size: int = 8,
                 frozen_bits: Optional[np.ndarray] = None,
                 use_crc: bool = False, crc_polynomial: str = "CRC-8"):
        assert N > 0 and (N & (N - 1)) == 0, "N must be a power of 2"
        assert 0 < K < N, "K must be in (0, N)"
        assert list_size >= 1

        self.N = N
        self.K = K
        self.L = list_size
        self.use_crc = use_crc
        self.crc_polynomial = crc_polynomial

        # 先复用 SCDecoder；后续在此基础上扩展真正的列表逻辑
        self._sc = SCDecoder(N, K, frozen_bits=frozen_bits)

    def decode(self, llr_input: np.ndarray) -> np.ndarray:
        # 目前直接调用 SC；L>1 时行为仍然与 SC 相同
        return self._sc.decode(llr_input)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"SCLDecoder(N={self.N}, K={self.K}, L={self.L}, use_crc={self.use_crc})"