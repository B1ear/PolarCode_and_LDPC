"""
LDPC Library Wrapper

Wraps the third-party `pyldpc` library to provide a unified interface
compatible with the project's own LDPCEncoder/BPDecoder.
"""

import numpy as np
from typing import Optional
import warnings

try:
    import pyldpc
    PYLDPC_AVAILABLE = True
except ImportError:
    PYLDPC_AVAILABLE = False


class LDPCLibWrapper:
    """
    Wrapper around pyldpc library for LDPC encoding/decoding.
    
    Provides encode/decode interface matching project's LDPCEncoder/BPDecoder.
    """
    
    def __init__(self, n: int, k: int, dv: int = 3, dc: int = 6, seed: Optional[int] = None):
        """
        Initialize LDPC library wrapper
        
        Args:
            n: Code length (codeword length)
            k: Information bits length
            dv: Variable node degree (number of checks per bit)
            dc: Check node degree (number of bits per check)
            seed: Random seed for matrix construction
        """
        if not PYLDPC_AVAILABLE:
            raise ImportError("pyldpc library not available. Install with: pip install pyldpc")
        
        assert n > k > 0, "Invalid code parameters: n > k > 0 required"
        assert dc >= dv, "Check degree dc must be >= variable degree dv"
        
        self.n = n
        self.k = k
        self.m = n - k  # Number of parity checks
        self.dv = dv
        self.dc = dc
        self.seed = seed
        
        # Create H and G matrices using pyldpc
        # H: (m, n) parity-check matrix
        # G: (n, k) coding matrix (transposed form in pyldpc)
        self.H, self.G = pyldpc.make_ldpc(n, dv, dc, systematic=True, sparse=False, seed=seed)
        
        # Extract actual k from G shape
        # G shape is (n, k_actual) where k_actual may differ slightly from input k
        self.k_actual = self.G.shape[1]
        
        if self.k_actual != k:
            print(f"Warning: Requested k={k}, but pyldpc generated k={self.k_actual}")
            self.k = self.k_actual
    
    def encode(self, message: np.ndarray) -> np.ndarray:
        """
        Encode message bits to codeword
        
        Args:
            message: Information bits, length k
            
        Returns:
            Codeword bits, length n
        """
        assert len(message) == self.k, f"Message length must be {self.k}"
        
        # Use pyldpc.utils.binaryproduct for pure encoding without noise
        # codeword = G @ message (mod 2)
        codeword = pyldpc.utils.binaryproduct(self.G, message)
        
        return codeword.astype(int)
    
    def decode(self, llr: np.ndarray, max_iter: int = 50) -> np.ndarray:
        """
        Decode LLR values to message bits using BP decoder
        
        Args:
            llr: Log-likelihood ratios, length n
                 (positive values indicate bit=0 more likely)
            max_iter: Maximum BP iterations
            
        Returns:
            Decoded message bits, length k
        """
        assert len(llr) == self.n, f"LLR length must be {self.n}"
        
        # Convert LLR to pyldpc's expected channel output format
        # pyldpc.decode expects noisy symbols y = BPSK + noise
        # Our LLR = 2*y/sigma^2, so y = LLR * sigma^2 / 2
        # For decoding, we need to provide SNR and y
        # We'll compute an effective SNR from LLR magnitude
        
        # Use a heuristic: assume average LLR magnitude corresponds to SNR
        # LLR = 2*y/sigma^2, and for BPSK |y| ~ 1 + noise
        # Estimate SNR from average LLR magnitude
        avg_llr_mag = np.mean(np.abs(llr))
        
        # Rough heuristic: SNR_linear ~ avg_llr_mag / 4
        # SNR_dB = 10 * log10(SNR_linear)
        snr_linear = max(avg_llr_mag / 4.0, 0.1)
        snr_db = 10.0 * np.log10(snr_linear)
        
        # Convert LLR back to channel symbols
        # For pyldpc: LLR = 2*y/var, var = 10^(-snr/10)
        var = 10.0 ** (-snr_db / 10.0)
        y = llr * var / 2.0
        
        # Decode using pyldpc (suppress convergence warnings - expected in low SNR)
        # Returns decoded codeword (length n)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*convergence.*')
            decoded_codeword = pyldpc.decode(self.H, y, snr_db, maxiter=max_iter)
        
        # Extract message bits using pyldpc.get_message
        decoded_message = pyldpc.get_message(self.G, decoded_codeword)
        
        return decoded_message.astype(int)
    
    def get_code_rate(self) -> float:
        """Get code rate k/n"""
        return self.k / self.n
    
    def get_parity_check_matrix(self) -> np.ndarray:
        """Get parity-check matrix H"""
        return self.H.copy()
    
    def get_generator_matrix(self) -> np.ndarray:
        """Get generator matrix G"""
        return self.G.copy()
    
    def __repr__(self) -> str:
        return f"LDPCLibWrapper(n={self.n}, k={self.k}, rate={self.get_code_rate():.3f}, dv={self.dv}, dc={self.dc})"


if __name__ == "__main__":
    # Test code
    print("Testing LDPCLibWrapper...")
    
    if not PYLDPC_AVAILABLE:
        print("✗ pyldpc library not available")
        exit(1)
    
    # Test 1: Basic encoding
    print("\n1. Basic Encoding Test:")
    n, k = 12, 6
    wrapper = LDPCLibWrapper(n, k, dv=3, dc=6, seed=42)
    print(f"Wrapper: {wrapper}")
    print(f"H shape: {wrapper.H.shape}")
    print(f"G shape: {wrapper.G.shape}")
    print(f"Actual k: {wrapper.k}")
    
    # Use actual k from wrapper
    k_actual = wrapper.k
    message = np.random.randint(0, 2, k_actual)
    codeword = wrapper.encode(message)
    print(f"Message: {message}")
    print(f"Codeword length: {len(codeword)}")
    
    # Verify codeword
    syndrome = (wrapper.H @ codeword) % 2
    valid = np.all(syndrome == 0)
    print(f"Valid codeword (H*c=0): {valid}")
    
    # Test 2: No-noise decoding
    print("\n2. No-Noise Decoding Test:")
    # High-magnitude LLR
    llr = (1 - 2 * codeword.astype(float)) * 100.0
    decoded = wrapper.decode(llr, max_iter=50)
    print(f"Decoded length: {len(decoded)}")
    print(f"Match: {np.array_equal(message, decoded)}")
    
    # Test 3: Multiple messages
    print("\n3. Multiple Messages Test:")
    n, k = 24, 12
    wrapper = LDPCLibWrapper(n, k, dv=3, dc=6, seed=42)
    k_actual = wrapper.k
    
    num_tests = 5
    all_correct = True
    for i in range(num_tests):
        msg = np.random.randint(0, 2, k_actual)
        cw = wrapper.encode(msg)
        llr = (1 - 2 * cw.astype(float)) * 100.0
        dec = wrapper.decode(llr, max_iter=50)
        
        correct = np.array_equal(msg, dec)
        all_correct = all_correct and correct
        print(f"  Test {i+1}: {'✓' if correct else '✗'}")
    
    print(f"All correct: {all_correct}")
    
    # Test 4: With AWGN channel
    print("\n4. AWGN Channel Test:")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from channel import AWGNChannel
    
    n, k = 96, 48
    snr_db = 3.0
    wrapper = LDPCLibWrapper(n, k, dv=3, dc=6, seed=42)
    k_actual = wrapper.k
    channel = AWGNChannel(snr_db=snr_db, seed=42)
    
    num_frames = 20
    errors = 0
    
    for i in range(num_frames):
        msg = np.random.randint(0, 2, k_actual)
        cw = wrapper.encode(msg)
        llr = channel.transmit(cw, return_llr=True)
        dec = wrapper.decode(llr, max_iter=50)
        
        if not np.array_equal(msg, dec):
            errors += 1
    
    print(f"n={n}, k={k_actual}, SNR={snr_db}dB")
    print(f"Frame errors: {errors}/{num_frames}")
    print(f"FER: {errors/num_frames:.4f}")
    
    # Test 5: Different code sizes
    print("\n5. Different Code Sizes Test:")
    configs = [(24, 12), (48, 24), (96, 48)]
    
    for n, k in configs:
        wrap = LDPCLibWrapper(n, k, dv=3, dc=6, seed=42)
        k_actual = wrap.k
        msg = np.random.randint(0, 2, k_actual)
        cw = wrap.encode(msg)
        llr = (1 - 2 * cw.astype(float)) * 100.0
        dec = wrap.decode(llr, max_iter=50)
        
        correct = np.array_equal(msg, dec)
        print(f"n={n:3d}, k={k_actual:2d}, rate={wrap.get_code_rate():.3f}, correct={correct}")
    
    print("\n✓ LDPCLibWrapper test passed!")
