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
    """Successive Cancellation List (SCL) Decoder.
    
    SCL decoder maintains multiple candidate paths during decoding,
    selecting the most likely path at the end.
    
    Args:
        N: Code length (must be power of 2)
        K: Information length
        list_size: Number of paths to maintain (L)
        frozen_bits: Optional frozen bit positions
        use_crc: Whether to use CRC for path selection
        crc_polynomial: CRC polynomial to use (if use_crc=True)
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
        self.n = int(np.log2(N))
        self.use_crc = use_crc
        self.crc_polynomial = crc_polynomial

        if frozen_bits is None:
            from .utils import generate_frozen_bits
            self.frozen_bits, self.info_bits = generate_frozen_bits(N, K)
        else:
            self.frozen_bits = np.array(frozen_bits, dtype=int)
            self.info_bits = np.setdiff1d(np.arange(N), self.frozen_bits)
        
        # Convert frozen_bits to set for O(1) lookup
        self.frozen_set = set(self.frozen_bits)
        
        # Path metrics (log-likelihood)
        self.path_metrics = np.full(self.L, -np.inf)
        
        # Each path maintains its own LLR and bit matrices
        self.L_paths = np.full((self.L, N, self.n + 1), np.nan, dtype=np.float64)
        self.B_paths = np.full((self.L, N, self.n + 1), np.nan, dtype=np.float64)
        
        # Active paths
        self.active_paths = np.zeros(self.L, dtype=bool)

    def decode(self, llr_input: np.ndarray) -> np.ndarray:
        """SCL decode with list decoding.
        
        Args:
            llr_input: Channel LLR values, length N
        
        Returns:
            Decoded information bits, length K
        """
        llr_input = np.asarray(llr_input, dtype=np.float64)
        assert llr_input.shape == (self.N,), f"expected LLR shape ({self.N},), got {llr_input.shape}"

        # Initialize: one active path with metric 0
        self.active_paths[:] = False
        self.active_paths[0] = True
        self.path_metrics[:] = -np.inf
        self.path_metrics[0] = 0.0
        
        # Initialize all paths with channel LLRs
        for l in range(self.L):
            self.L_paths[l, :, 0] = llr_input
        
        # Decode bits in bit-reversed order
        for i in range(self.N):
            l = bit_reverse(i, self.n)
            
            if l in self.frozen_set:
                # Frozen bit: force to 0, no path split
                self._decode_frozen_bit(l)
            else:
                # Information bit: split paths
                self._decode_info_bit(l)
        
        # Select best path
        best_path_idx = np.argmax(self.path_metrics)
        u_decoded = self.B_paths[best_path_idx, :, self.n].astype(int)
        
        return u_decoded[self.info_bits]

    def _decode_frozen_bit(self, l: int):
        """Decode a frozen bit (set to 0 for all active paths)."""
        for path_idx in range(self.L):
            if not self.active_paths[path_idx]:
                continue
            
            # Update LLRs for this path
            self._update_llrs_for_path(path_idx, l)
            
            # Set frozen bit to 0
            self.B_paths[path_idx, l, self.n] = 0
            
            # Update path metric
            llr_val = self.L_paths[path_idx, l, self.n]
            self.path_metrics[path_idx] += self._log_likelihood(llr_val, 0)
            
            # Propagate bit decision
            self._update_bits_for_path(path_idx, l)

    def _decode_info_bit(self, l: int):
        """Decode an information bit (split paths for 0 and 1)."""
        # Find active paths
        active_indices = np.where(self.active_paths)[0]
        n_active = len(active_indices)
        
        # Calculate metrics for both bit choices
        path_metrics_0 = []
        path_metrics_1 = []
        
        for path_idx in active_indices:
            # Update LLRs
            self._update_llrs_for_path(path_idx, l)
            llr_val = self.L_paths[path_idx, l, self.n]
            
            # Metrics for bit=0 and bit=1
            metric_0 = self.path_metrics[path_idx] + self._log_likelihood(llr_val, 0)
            metric_1 = self.path_metrics[path_idx] + self._log_likelihood(llr_val, 1)
            
            path_metrics_0.append((metric_0, path_idx, 0))
            path_metrics_1.append((metric_1, path_idx, 1))
        
        # Combine and sort all candidate paths
        all_candidates = path_metrics_0 + path_metrics_1
        all_candidates.sort(key=lambda x: x[0], reverse=True)  # Sort by metric (descending)
        
        # Select top L paths
        n_survive = min(len(all_candidates), self.L)
        survivors = all_candidates[:n_survive]
        
        # Store old path states
        old_L_paths = self.L_paths.copy()
        old_B_paths = self.B_paths.copy()
        old_metrics = self.path_metrics.copy()
        
        # Reset active paths
        self.active_paths[:] = False
        self.path_metrics[:] = -np.inf
        
        # Update survivors
        for new_idx, (metric, old_idx, bit) in enumerate(survivors):
            if new_idx >= self.L:
                break
            
            # Copy path state
            self.L_paths[new_idx] = old_L_paths[old_idx].copy()
            self.B_paths[new_idx] = old_B_paths[old_idx].copy()
            
            # Set bit decision
            self.B_paths[new_idx, l, self.n] = bit
            
            # Update metric
            self.path_metrics[new_idx] = metric
            self.active_paths[new_idx] = True
            
            # Propagate bit decision
            self._update_bits_for_path(new_idx, l)

    def _update_llrs_for_path(self, path_idx: int, l: int):
        """Update LLRs for a specific path."""
        for s in range(self.n - self._active_llr_level(l, self.n), self.n):
            block_size = int(2 ** (s + 1))
            branch_size = int(block_size / 2)
            
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:  # upper branch
                    top_llr = self.L_paths[path_idx, j, s]
                    btm_llr = self.L_paths[path_idx, j + branch_size, s]
                    self.L_paths[path_idx, j, s + 1] = self._upper_llr(top_llr, btm_llr)
                else:  # lower branch
                    btm_llr = self.L_paths[path_idx, j, s]
                    top_llr = self.L_paths[path_idx, j - branch_size, s]
                    top_bit = self.B_paths[path_idx, j - branch_size, s + 1]
                    self.L_paths[path_idx, j, s + 1] = self._lower_llr(btm_llr, top_llr, top_bit)

    def _update_bits_for_path(self, path_idx: int, l: int):
        """Update bits for a specific path."""
        if l < self.N / 2:
            return
        
        for s in range(self.n, self.n - self._active_bit_level(l, self.n), -1):
            block_size = int(2 ** s)
            branch_size = int(block_size / 2)
            
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:  # lower branch
                    self.B_paths[path_idx, j - branch_size, s - 1] = (
                        int(self.B_paths[path_idx, j, s]) ^ int(self.B_paths[path_idx, j - branch_size, s])
                    )
                    self.B_paths[path_idx, j, s - 1] = self.B_paths[path_idx, j, s]

    def _log_likelihood(self, llr: float, bit: int) -> float:
        """Calculate log-likelihood contribution (to maximize).
        
        For LLR = log(P(bit=0)/P(bit=1)):
        We want to compute log P(bit|LLR)
        
        P(bit=0|LLR) = e^LLR / (1 + e^LLR) = 1 / (1 + e^-LLR)
        P(bit=1|LLR) = 1 / (1 + e^LLR)
        
        log P(bit=0|LLR) = -log(1 + e^-LLR)
        log P(bit=1|LLR) = -log(1 + e^LLR)
        
        Using numerically stable computation:
        -log(1 + e^x) = -log(e^max(0,x) * (e^-max(0,x) + e^(x-max(0,x))))
                      = -max(0, x) - log1p(e^(-|x|))
        """
        # Numerically stable computation
        if bit == 0:
            # log P(bit=0) = -log(1 + e^-LLR)
            if llr >= 0:
                # For positive LLR, e^-LLR is small
                return -np.log1p(np.exp(-llr))
            else:
                # For negative LLR, rewrite as: -log(1 + e^-LLR) = -log(e^-LLR(e^LLR + 1)) = LLR - log(1 + e^LLR)
                return llr - np.log1p(np.exp(llr))
        else:
            # log P(bit=1) = -log(1 + e^LLR)
            if llr >= 0:
                # For positive LLR, rewrite: -log(1 + e^LLR) = -log(e^LLR(e^-LLR + 1)) = -LLR - log(1 + e^-LLR)
                return -llr - np.log1p(np.exp(-llr))
            else:
                # For negative LLR, e^LLR is small
                return -np.log1p(np.exp(llr))

    def _upper_llr(self, l1: float, l2: float) -> float:
        """Upper branch LLR update (f function) - min-sum approximation."""
        return np.sign(l1) * np.sign(l2) * min(abs(l1), abs(l2))

    def _lower_llr(self, l1: float, l2: float, b: float) -> float:
        """Lower branch LLR update (g function)."""
        if b == 0:
            return l1 + l2
        else:
            return l1 - l2

    def _active_llr_level(self, i: int, n: int) -> int:
        """Find the first 1 in binary expansion of i."""
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
        """Find the first 0 in binary expansion of i."""
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
        return f"SCLDecoder(N={self.N}, K={self.K}, L={self.L}, use_crc={self.use_crc})"
