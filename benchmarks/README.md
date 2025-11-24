# Benchmark System

Complete benchmark suite for Polar Code and LDPC performance evaluation.

## Modules

### 1. BER Simulation (`ber_simulation.py`)
Performs Bit Error Rate (BER) and Frame Error Rate (FER) simulation over SNR range.

**Features:**
- SNR sweep testing
- Early stopping (max errors threshold)
- Support for both self-implemented and third-party library codes
- Generates BER/FER curves
- Saves results as JSON and PNG plots

### 2. Throughput Test (`throughput_test.py`)
Measures encoding and decoding throughput in Mbps.

**Features:**
- Encoding throughput measurement
- Decoding throughput measurement
- End-to-end throughput
- Warm-up runs to eliminate startup effects

### 3. Complexity Analysis (`complexity_analysis.py`)
Estimates computational complexity and memory usage.

**Features:**
- Operation counts (XOR, multiply, etc.)
- Memory usage estimates
- Theoretical complexity formulas
- Comparative plots

## Usage

### Quick Start

Run all benchmarks with default settings:
```bash
python benchmarks/run_benchmark.py
```

### Custom Configuration

**Short BER test:**
```bash
python benchmarks/run_benchmark.py \
    --snr-range "0:5:1" \
    --num-frames 100 \
    --max-errors 50
```

**Skip specific tests:**
```bash
# Only BER simulation
python benchmarks/run_benchmark.py \
    --skip-throughput \
    --skip-complexity

# Only throughput
python benchmarks/run_benchmark.py \
    --skip-ber \
    --skip-complexity
```

**Custom configurations:**
```bash
python benchmarks/run_benchmark.py \
    --polar-config config/polar_config.yaml \
    --ldpc-config config/ldpc_config.yaml \
    --output-dir my_results
```

**Enable third-party library comparison:**
```bash
python benchmarks/run_benchmark.py \
    --use-third-party \
    --snr-range "0:5:1" \
    --num-frames 100
```

### Run Individual Modules

Each module can be run standalone for testing:

```bash
# BER simulation
python benchmarks/ber_simulation.py

# Throughput test
python benchmarks/throughput_test.py

# Complexity analysis
python benchmarks/complexity_analysis.py
```

## Command-Line Options

```
--snr-range START:STOP:STEP    SNR range for BER simulation (default: -2:6:0.5)
--num-frames N                  Max frames per SNR point (default: 1000)
--max-errors N                  Stop after N frame errors (default: 100)
--polar-config FILE             Polar configuration YAML (default: config/polar_config.yaml)
--ldpc-config FILE              LDPC configuration YAML (default: config/ldpc_config.yaml)
--output-dir DIR                Output directory (default: results)
--skip-ber                      Skip BER simulation
--skip-throughput               Skip throughput test
--skip-complexity               Skip complexity analysis
--use-third-party               Enable third-party library comparison
--throughput-iterations N       Number of iterations for throughput test (default: 100)
```

## Output

Results are saved in the specified output directory (default: `results/`):

```
results/
├── figures/
│   ├── ber_curves.png          # BER vs SNR
│   ├── fer_curves.png          # FER vs SNR
│   └── complexity_comparison.png
└── data/
    ├── ber_simulation_results.json
    ├── throughput_results.json
    ├── complexity_results.json
    └── benchmark_results.json  # Combined results
```

## Configuration Files

Edit `config/polar_config.yaml` and `config/ldpc_config.yaml` to customize code parameters:

**Polar Config:**
```yaml
encoding:
  N: 1024        # Code length (power of 2)
  K: 512         # Information bits

construction:
  method: "ga"   # Construction method
  design_snr_db: 2.0
```

**LDPC Config:**
```yaml
encoding:
  n: 504         # Code length
  k: 252         # Information bits
  dv: 3          # Variable node degree
  dc: 6          # Check node degree

decoding:
  max_iterations: 50
  algorithm: "bp"  # or "ms" for min-sum
```

## Example Results

**Typical BER Performance (SNR = 2dB):**
- Polar Code: BER ≈ 0.33, FER = 1.0
- LDPC: BER ≈ 0.03, FER = 0.91

**Throughput (N=128, K=64):**
- Polar Encoding: ~0.18 Mbps
- Polar Decoding: ~0.19 Mbps
- LDPC Encoding: ~0.01 Mbps
- LDPC Decoding: ~0.00 Mbps (slower due to BP iterations)

**Complexity (N=128, K=64):**
- Polar Encoding: 896 XOR operations, O(N log N)
- Polar Decoding: 1,024 operations, O(N log N)
- LDPC Encoding: ~3,600 operations, O(m × k)
- LDPC Decoding: ~90,000 operations (50 iterations), O(I × edges)

## Notes

### Performance Considerations

- **LDPC decoding is slow**: BP decoder with 50 iterations takes ~20-30 seconds per 1000 frames in Python
  - For faster testing, use `--throughput-iterations 50` (default is now 100)
  - Or skip throughput test with `--skip-throughput`
  
- **Quick testing**: Use the quick benchmark script:
  ```bash
  python quick_benchmark.py  # ~3-5 minutes, includes third-party comparison
  ```

- **LDPC encoding warnings**: "Could not create systematic generator matrix" is expected for some parameters
  - Uses direct solving method (slower but correct)
  
- **Throughput**: Python overhead is significant; C/C++ implementations would be 10-100× faster

- **Early stopping**: BER simulation stops after `--max-errors` frame errors, reducing test time at high SNR

- **Publication quality**: For research papers, use:
  ```bash
  python benchmarks/run_benchmark.py \
      --snr-range "-2:6:0.5" \
      --num-frames 10000 \
      --max-errors 200 \
      --throughput-iterations 1000
  ```

## Third-Party Library Comparison

### Installation

First, install the required third-party libraries:

```bash
pip install pyldpc py-polar-codes
```

### Usage

Enable third-party library comparison using the `--use-third-party` flag:

```bash
python benchmarks/run_benchmark.py --use-third-party
```

This will:
- Run both self-implemented and third-party library versions
- Generate plots with 4 curves: Polar (Self), Polar (Library), LDPC (Self), LDPC (Library)
- Compare performance and correctness

**Note:** Third-party libraries typically show much better performance because:
- `polarcodes` uses a proper soft-decision SC decoder (vs our hard-decision)
- `pyldpc` has optimized BP implementation with numba acceleration
- Both libraries are well-tested and production-ready

This comparison is useful for:
- Validating correctness of self-implementation
- Understanding performance gaps
- Benchmarking against state-of-the-art
