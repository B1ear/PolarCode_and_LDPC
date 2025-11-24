"""
Polar Code Library Wrapper

Wraps the third-party `polarcodes` library (py-polar-codes) to provide
a unified interface compatible with the project's own PolarEncoder/SCDecoder.
"""

import numpy as np
from typing import Optional

try:
    from polarcodes import PolarCode, Construct, Encode, Decode
    POLARCODES_AVAILABLE = True
except ImportError:
    POLARCODES_AVAILABLE = False


class PolarLibWrapper:
    """
    Wrapper around polarcodes library for Polar encoding/decoding.
    
    Provides encode/decode interface matching project's PolarEncoder/SCDecoder.
    """
    
    def __init__(self, N: int, K: int, design_snr_db: float = 2.0):
        """
        Initialize Polar library wrapper
        
        Args:
            N: Code length (must be power of 2)
            K: Information bits length
            design_snr_db: Design SNR for code construction (dB)
        """
        if not POLARCODES_AVAILABLE:
            raise ImportError("polarcodes library not available. Install with: pip install py-polar-codes")
        
        assert N > 0 and (N & (N - 1)) == 0, "N must be a power of 2"
        assert 0 < K < N, "K must be in (0, N)"
        
        self.N = N
        self.K = K
        self.design_snr_db = design_snr_db
        
        # Create PolarCode instance and construct frozen set
        self.pc = PolarCode(N, K)
        Construct(self.pc, design_SNR=design_snr_db)
        
        # Store frozen and info bit positions for reference
        self.frozen_bits = self.pc.frozen
        self.info_bits = np.setdiff1d(np.arange(N), self.frozen_bits)
    
    def encode(self, message: np.ndarray) -> np.ndarray:
        """
        Encode message bits to codeword
        
        Args:
            message: Information bits, length K
            
        Returns:
            Codeword bits, length N
        """
        assert len(message) == self.K, f"Message length must be {self.K}"
        
        # Set message, encode, and get codeword
        self.pc.set_message(message.astype(int))
        Encode(self.pc)
        codeword = self.pc.get_codeword()
        
        return codeword.astype(int)
    
    def decode(self, llr: np.ndarray) -> np.ndarray:
        """
        Decode LLR values to message bits using SC decoder
        
        Args:
            llr: Log-likelihood ratios, length N
                 (positive values indicate bit=0 more likely)
            
        Returns:
            Decoded message bits, length K
        """
        assert len(llr) == self.N, f"LLR length must be {self.N}"
        
        # Set likelihoods and decode
        self.pc.likelihoods = llr.astype(float)
        Decode(self.pc, decoder_name='scd')
        
        # Get decoded message
        decoded_message = self.pc.message_received
        
        return decoded_message.astype(int)
    
    def get_code_rate(self) -> float:
        """Get code rate K/N"""
        return self.K / self.N
    
    def get_frozen_bits_positions(self) -> np.ndarray:
        """Get frozen bit positions"""
        return self.frozen_bits.copy()
    
    def get_info_bits_positions(self) -> np.ndarray:
        """Get information bit positions"""
        return self.info_bits.copy()
    
    def __repr__(self) -> str:
        return f"PolarLibWrapper(N={self.N}, K={self.K}, rate={self.get_code_rate():.3f}, design_SNR={self.design_snr_db}dB)"


if __name__ == "__main__":
    # Test code
    print("Testing PolarLibWrapper...")
    
    if not POLARCODES_AVAILABLE:
        print("✗ polarcodes library not available")
        exit(1)
    
    # Test 1: Basic encoding
    print("\n1. Basic Encoding Test:")
    N, K = 8, 4
    wrapper = PolarLibWrapper(N, K)
    print(f"Wrapper: {wrapper}")
    print(f"Frozen bits: {wrapper.get_frozen_bits_positions()}")
    print(f"Info bits: {wrapper.get_info_bits_positions()}")
    
    message = np.array([1, 0, 1, 1])
    codeword = wrapper.encode(message)
    print(f"Message: {message}")
    print(f"Codeword: {codeword}")
    
    # Test 2: No-noise decoding
    print("\n2. No-Noise Decoding Test:")
    # High-magnitude LLR
    llr = (1 - 2 * codeword.astype(float)) * 1000.0
    decoded = wrapper.decode(llr)
    print(f"Decoded: {decoded}")
    print(f"Match: {np.array_equal(message, decoded)}")
    
    # Test 3: Multiple messages
    print("\n3. Multiple Messages Test:")
    N, K = 16, 8
    wrapper = PolarLibWrapper(N, K)
    
    num_tests = 5
    all_correct = True
    for i in range(num_tests):
        msg = np.random.randint(0, 2, K)
        cw = wrapper.encode(msg)
        llr = (1 - 2 * cw.astype(float)) * 100.0
        dec = wrapper.decode(llr)
        
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
    
    N, K = 128, 64
    snr_db = 4.0
    wrapper = PolarLibWrapper(N, K, design_snr_db=2.0)
    channel = AWGNChannel(snr_db=snr_db, seed=42)
    
    num_frames = 20
    errors = 0
    
    for i in range(num_frames):
        msg = np.random.randint(0, 2, K)
        cw = wrapper.encode(msg)
        llr = channel.transmit(cw, return_llr=True)
        dec = wrapper.decode(llr)
        
        if not np.array_equal(msg, dec):
            errors += 1
    
    print(f"N={N}, K={K}, SNR={snr_db}dB")
    print(f"Frame errors: {errors}/{num_frames}")
    print(f"FER: {errors/num_frames:.4f}")
    
    # Test 5: Different code sizes
    print("\n5. Different Code Sizes Test:")
    configs = [(8, 4), (16, 8), (32, 16), (64, 32)]
    
    for N, K in configs:
        wrap = PolarLibWrapper(N, K)
        msg = np.random.randint(0, 2, K)
        cw = wrap.encode(msg)
        llr = (1 - 2 * cw.astype(float)) * 100.0
        dec = wrap.decode(llr)
        
        correct = np.array_equal(msg, dec)
        print(f"N={N:3d}, K={K:2d}, rate={wrap.get_code_rate():.3f}, correct={correct}")
    
    print("\n✓ PolarLibWrapper test passed!")
