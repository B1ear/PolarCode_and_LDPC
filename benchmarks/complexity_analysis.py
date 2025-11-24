"""
Complexity Analysis

Estimates computational complexity (operations count, memory usage) for Polar and LDPC codes.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.visualization import save_results


def analyze_complexity(
    polar_config: Dict,
    ldpc_config: Dict,
    output_dir: Path
) -> Dict:
    """
    Analyze computational complexity of Polar and LDPC implementations
    
    Args:
        polar_config: Polar code configuration
        ldpc_config: LDPC configuration
        output_dir: Output directory for results
        
    Returns:
        Dictionary containing complexity estimates
    """
    print(f"\n{'='*60}")
    print("Complexity Analysis")
    print(f"{'='*60}")
    
    results = {
        'polar': {},
        'ldpc': {}
    }
    
    # Analyze Polar Code complexity
    print(f"\n{'-'*60}")
    print("Analyzing Polar Code Complexity")
    print(f"{'-'*60}")
    
    polar_complexity = analyze_polar_complexity(polar_config)
    results['polar'] = polar_complexity
    
    # Analyze LDPC complexity
    print(f"\n{'-'*60}")
    print("Analyzing LDPC Complexity")
    print(f"{'-'*60}")
    
    ldpc_complexity = analyze_ldpc_complexity(ldpc_config)
    results['ldpc'] = ldpc_complexity
    
    # Print summary
    print(f"\n{'='*60}")
    print("Complexity Summary")
    print(f"{'='*60}")
    
    print(f"\nPolar Code (N={polar_complexity['N']}, K={polar_complexity['K']}):")
    print(f"  Encoding Operations:  {polar_complexity['encoding_ops']:,}")
    print(f"  Decoding Operations:  {polar_complexity['decoding_ops']:,}")
    print(f"  Encoding Memory:      {polar_complexity['encoding_memory']:,} bits")
    print(f"  Decoding Memory:      {polar_complexity['decoding_memory']:,} bits")
    
    print(f"\nLDPC (n={ldpc_complexity['n']}, k={ldpc_complexity['k']}):")
    print(f"  Encoding Operations:  {ldpc_complexity['encoding_ops']:,}")
    print(f"  Decoding Operations:  {ldpc_complexity['decoding_ops']:,}")
    print(f"  Encoding Memory:      {ldpc_complexity['encoding_memory']:,} bits")
    print(f"  Decoding Memory:      {ldpc_complexity['decoding_memory']:,} bits")
    
    # Plot comparison
    plot_complexity_comparison(results, output_dir)
    
    # Save results
    save_results(results, output_dir / "data" / "complexity_results.json")
    
    return results


def analyze_polar_complexity(config: Dict) -> Dict:
    """Analyze Polar code complexity"""
    
    N = config['encoding']['N']
    K = config['encoding']['K']
    n = int(np.log2(N))  # log2(N)
    
    print(f"Polar: N={N}, K={K}, n=log2(N)={n}")
    
    # Encoding Complexity
    # Polar encoding: O(N log N) XOR operations
    # Each stage has N/2 butterfly operations (2 XORs each)
    # There are log2(N) stages
    encoding_ops = N * n  # Total XOR operations
    
    print(f"\nEncoding Complexity:")
    print(f"  Algorithm: Successive Butterfly (iterative)")
    print(f"  Stages: {n}")
    print(f"  Operations per stage: {N} XORs")
    print(f"  Total XOR operations: {encoding_ops}")
    print(f"  Time Complexity: O(N log N)")
    
    # Encoding Memory
    # Need to store: input vector (N), output vector (N)
    encoding_memory = 2 * N  # bits
    
    print(f"  Memory: {encoding_memory} bits ({encoding_memory/8:.0f} bytes)")
    
    # Decoding Complexity (SC Decoder - hard decision + inverse transform)
    # Hard decision: N comparisons
    # Inverse transform: N log N XORs (same as encoding)
    decoding_ops_hard = N  # comparisons
    decoding_ops_transform = N * n  # XORs for inverse transform
    decoding_ops = decoding_ops_hard + decoding_ops_transform
    
    print(f"\nDecoding Complexity (SC - Hard Decision + Inverse Transform):")
    print(f"  Hard decision: {decoding_ops_hard} comparisons")
    print(f"  Inverse transform: {decoding_ops_transform} XORs")
    print(f"  Total operations: {decoding_ops}")
    print(f"  Time Complexity: O(N log N)")
    
    # Decoding Memory
    # Need: received LLRs (N), decoded bits (N)
    decoding_memory = 2 * N  # bits
    
    print(f"  Memory: {decoding_memory} bits ({decoding_memory/8:.0f} bytes)")
    
    # SCL Decoder complexity (for reference, even though we don't use it)
    L = 8  # typical list size
    scl_ops = L * N * n  # L times SC complexity
    scl_memory = L * 2 * N  # L copies of SC memory
    
    print(f"\nSCL Decoder Complexity (L={L}, for reference):")
    print(f"  Operations: {scl_ops:,} (L × SC complexity)")
    print(f"  Memory: {scl_memory:,} bits ({scl_memory/8:.0f} bytes)")
    
    return {
        'N': N,
        'K': K,
        'n': n,
        'rate': K / N,
        'encoding_ops': encoding_ops,
        'encoding_memory': encoding_memory,
        'encoding_complexity': f"O(N log N) = O({N} × {n})",
        'decoding_ops': decoding_ops,
        'decoding_memory': decoding_memory,
        'decoding_complexity': f"O(N log N) = O({N} × {n})",
        'scl_ops': scl_ops,
        'scl_memory': scl_memory
    }


def analyze_ldpc_complexity(config: Dict) -> Dict:
    """Analyze LDPC complexity"""
    
    n = config['encoding']['n']
    k = config['encoding']['k']
    m = n - k
    dv = config['encoding'].get('dv', 3)
    dc = config['encoding'].get('dc', 6)
    max_iter = config['decoding'].get('max_iterations', 50)
    
    print(f"LDPC: n={n}, k={k}, m={m}, dv={dv}, dc={dc}")
    
    # Encoding Complexity
    # For systematic code: O(m * k) operations
    # Need to compute parity bits: p = H2^-1 * H1 * m
    encoding_ops = m * k  # Matrix-vector multiplication
    
    print(f"\nEncoding Complexity:")
    print(f"  Algorithm: Systematic encoding (H * c = 0)")
    print(f"  Matrix-vector mult: {m} × {k}")
    print(f"  Operations: ~{encoding_ops}")
    print(f"  Time Complexity: O(m × k)")
    
    # Encoding Memory
    # Need: message (k), codeword (n), H matrix (m × n)
    encoding_memory = k + n + (m * n)
    
    print(f"  Memory: ~{encoding_memory} bits ({encoding_memory/8:.0f} bytes)")
    print(f"    Message: {k} bits")
    print(f"    Codeword: {n} bits")
    print(f"    H matrix: {m}×{n} = {m*n} bits")
    
    # Decoding Complexity (BP Decoder)
    # Per iteration:
    #   - Variable node update: O(n × dv) operations
    #   - Check node update: O(m × dc) operations
    # Total: O(max_iter × (n × dv + m × dc))
    
    ops_per_var_node = dv * 2  # Sum and subtract for each edge
    ops_per_check_node = dc * 3  # Product, min, sign for each edge
    
    ops_per_iter = n * ops_per_var_node + m * ops_per_check_node
    decoding_ops = max_iter * ops_per_iter
    
    print(f"\nDecoding Complexity (BP Decoder):")
    print(f"  Algorithm: Belief Propagation (Sum-Product)")
    print(f"  Max iterations: {max_iter}")
    print(f"  Operations per iteration:")
    print(f"    Variable nodes: {n} × {ops_per_var_node} = {n * ops_per_var_node}")
    print(f"    Check nodes: {m} × {ops_per_check_node} = {m * ops_per_check_node}")
    print(f"    Total per iter: {ops_per_iter}")
    print(f"  Total operations: ~{decoding_ops:,}")
    print(f"  Time Complexity: O(I × (n × dv + m × dc))")
    
    # Decoding Memory
    # Need: LLRs (n), messages v2c (n × dv), messages c2v (m × dc)
    num_edges = n * dv  # = m * dc
    decoding_memory = n + 2 * num_edges  # LLRs + two message arrays
    
    print(f"  Memory: ~{decoding_memory} bits ({decoding_memory/8:.0f} bytes)")
    print(f"    LLRs: {n} values")
    print(f"    Messages: 2 × {num_edges} = {2*num_edges} values")
    
    return {
        'n': n,
        'k': k,
        'm': m,
        'dv': dv,
        'dc': dc,
        'max_iter': max_iter,
        'rate': k / n,
        'encoding_ops': encoding_ops,
        'encoding_memory': encoding_memory,
        'encoding_complexity': f"O(m × k) = O({m} × {k})",
        'decoding_ops': decoding_ops,
        'decoding_memory': decoding_memory,
        'decoding_complexity': f"O(I × (n×dv + m×dc)) = O({max_iter} × {ops_per_iter})"
    }


def plot_complexity_comparison(results: Dict, output_dir: Path):
    """Plot complexity comparison"""
    
    polar = results['polar']
    ldpc = results['ldpc']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Operations comparison
    ax = axes[0]
    
    categories = ['Encoding', 'Decoding']
    polar_ops = [polar['encoding_ops'], polar['decoding_ops']]
    ldpc_ops = [ldpc['encoding_ops'], ldpc['decoding_ops']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, polar_ops, width, label='Polar', alpha=0.8)
    ax.bar(x + width/2, ldpc_ops, width, label='LDPC', alpha=0.8)
    
    ax.set_xlabel('Operation', fontsize=12)
    ax.set_ylabel('Number of Operations', fontsize=12)
    ax.set_title('Computational Complexity', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(polar_ops):
        ax.text(i - width/2, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(ldpc_ops):
        ax.text(i + width/2, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
    
    # Memory comparison
    ax = axes[1]
    
    polar_mem = [polar['encoding_memory'], polar['decoding_memory']]
    ldpc_mem = [ldpc['encoding_memory'], ldpc['decoding_memory']]
    
    ax.bar(x - width/2, polar_mem, width, label='Polar', alpha=0.8)
    ax.bar(x + width/2, ldpc_mem, width, label='LDPC', alpha=0.8)
    
    ax.set_xlabel('Operation', fontsize=12)
    ax.set_ylabel('Memory (bits)', fontsize=12)
    ax.set_title('Memory Usage', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(polar_mem):
        ax.text(i - width/2, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(ldpc_mem):
        ax.text(i + width/2, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    complexity_path = output_dir / "figures" / "complexity_comparison.png"
    plt.savefig(complexity_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved complexity plot: {complexity_path}")
    plt.close()


if __name__ == "__main__":
    # Test complexity analysis
    print("Testing Complexity Analysis Module...")
    
    polar_config = {
        'encoding': {'N': 128, 'K': 64},
        'construction': {'design_snr_db': 2.0}
    }
    
    ldpc_config = {
        'encoding': {'n': 120, 'k': 60, 'dv': 3, 'dc': 6},
        'decoding': {'max_iterations': 50}
    }
    
    # Use absolute path relative to this file
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    
    results = analyze_complexity(
        polar_config=polar_config,
        ldpc_config=ldpc_config,
        output_dir=output_dir
    )
    
    print("\n✓ Complexity analysis test passed!")
