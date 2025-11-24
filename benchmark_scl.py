"""
Comprehensive performance benchmark for SCL decoder.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.polar.encoder import PolarEncoder
from src.polar.decoder import SCDecoder, SCLDecoder
from src.channel.awgn import AWGNChannel


def ber_simulation(N, K, snr_range, list_sizes, n_trials=500):
    """Run BER simulation for different list sizes."""
    print(f"=== BER Simulation: N={N}, K={K}, trials={n_trials} ===\n")
    
    encoder = PolarEncoder(N, K)
    
    # Prepare decoders
    sc_decoder = SCDecoder(N, K, frozen_bits=encoder.frozen_bits)
    scl_decoders = {
        L: SCLDecoder(N, K, list_size=L, frozen_bits=encoder.frozen_bits)
        for L in list_sizes
    }
    
    # Store results
    ber_sc = []
    ber_scl = {L: [] for L in list_sizes}
    
    for snr_db in snr_range:
        print(f"SNR = {snr_db} dB...")
        channel = AWGNChannel(snr_db)
        
        # Count bit errors
        sc_bit_errors = 0
        scl_bit_errors = {L: 0 for L in list_sizes}
        total_bits = n_trials * K
        
        np.random.seed(42 + int(snr_db * 10))
        
        for _ in range(n_trials):
            # Generate and encode
            message = np.random.randint(0, 2, K)
            codeword = encoder.encode(message)
            
            # Transmit
            llr = channel.transmit(codeword, return_llr=True)
            
            # SC decode
            decoded_sc = sc_decoder.decode(llr)
            sc_bit_errors += np.sum(decoded_sc != message)
            
            # SCL decode
            for L, decoder in scl_decoders.items():
                decoded_scl = decoder.decode(llr)
                scl_bit_errors[L] += np.sum(decoded_scl != message)
        
        # Calculate BER
        ber_sc.append(sc_bit_errors / total_bits)
        for L in list_sizes:
            ber_scl[L].append(scl_bit_errors[L] / total_bits)
        
        print(f"  SC BER: {ber_sc[-1]:.6f}")
        for L in list_sizes:
            print(f"  SCL(L={L:2d}) BER: {ber_scl[L][-1]:.6f}")
        print()
    
    return ber_sc, ber_scl


def frame_error_rate_simulation(N, K, snr_range, list_sizes, n_trials=500):
    """Run FER simulation for different list sizes."""
    print(f"=== FER Simulation: N={N}, K={K}, trials={n_trials} ===\n")
    
    encoder = PolarEncoder(N, K)
    
    # Prepare decoders
    sc_decoder = SCDecoder(N, K, frozen_bits=encoder.frozen_bits)
    scl_decoders = {
        L: SCLDecoder(N, K, list_size=L, frozen_bits=encoder.frozen_bits)
        for L in list_sizes
    }
    
    # Store results
    fer_sc = []
    fer_scl = {L: [] for L in list_sizes}
    
    for snr_db in snr_range:
        print(f"SNR = {snr_db} dB...")
        channel = AWGNChannel(snr_db)
        
        # Count frame errors
        sc_frame_errors = 0
        scl_frame_errors = {L: 0 for L in list_sizes}
        
        np.random.seed(123 + int(snr_db * 10))
        
        for _ in range(n_trials):
            # Generate and encode
            message = np.random.randint(0, 2, K)
            codeword = encoder.encode(message)
            
            # Transmit
            llr = channel.transmit(codeword, return_llr=True)
            
            # SC decode
            decoded_sc = sc_decoder.decode(llr)
            if not np.array_equal(decoded_sc, message):
                sc_frame_errors += 1
            
            # SCL decode
            for L, decoder in scl_decoders.items():
                decoded_scl = decoder.decode(llr)
                if not np.array_equal(decoded_scl, message):
                    scl_frame_errors[L] += 1
        
        # Calculate FER
        fer_sc.append(sc_frame_errors / n_trials)
        for L in list_sizes:
            fer_scl[L].append(scl_frame_errors[L] / n_trials)
        
        print(f"  SC FER: {fer_sc[-1]:.6f}")
        for L in list_sizes:
            print(f"  SCL(L={L:2d}) FER: {fer_scl[L][-1]:.6f}")
        print()
    
    return fer_sc, fer_scl


if __name__ == "__main__":
    # Configuration
    N, K = 128, 64
    snr_range = np.arange(0.0, 3.5, 0.5)
    list_sizes = [1, 2, 4, 8]
    n_trials = 200
    
    print("=" * 60)
    print("SCL Decoder Performance Benchmark")
    print("=" * 60)
    print()
    
    # Run FER simulation
    fer_sc, fer_scl = frame_error_rate_simulation(N, K, snr_range, list_sizes, n_trials)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary: Frame Error Rate")
    print("=" * 60)
    print(f"\n{'SNR (dB)':<10}", end="")
    print(f"{'SC':<12}", end="")
    for L in list_sizes:
        print(f"{'SCL(L=' + str(L) + ')':<12}", end="")
    print()
    print("-" * 60)
    
    for i, snr in enumerate(snr_range):
        print(f"{snr:<10.1f}", end="")
        print(f"{fer_sc[i]:<12.6f}", end="")
        for L in list_sizes:
            print(f"{fer_scl[L][i]:<12.6f}", end="")
        print()
    
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)
