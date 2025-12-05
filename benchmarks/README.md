# 性能测试系统

Polar码和LDPC性能评估的完整测试套件。

## 模块说明

### 1. BER仿真 (`ber_simulation.py`)
在SNR范围内执行误码率(BER)和帧错误率(FER)仿真。

**功能特性：**
- SNR扫描测试
- 早停机制（最大错误阈值）
- 支持自实现和第三方库代码
- 生成BER/FER曲线
- 保存结果为JSON和PNG图表

### 2. 吞吐量测试 (`throughput_test.py`)
测量编码和解码吞吐量（Mbps）。

**功能特性：**
- 编码吞吐量测量
- 解码吞吐量测量
- 端到端吞吐量
- 预热运行以消除启动效应

### 3. 复杂度分析 (`complexity_analysis.py`)
估算计算复杂度和内存使用。

**功能特性：**
- 操作计数（异或、乘法等）
- 内存使用估算
- 理论复杂度公式
- 对比图表

## 使用方法

### 快速开始

使用默认设置运行所有测试：
```bash
python benchmarks/run_benchmark.py
```

### 自定义配置

**快速BER测试：**
```bash
python benchmarks/run_benchmark.py \
    --snr-range "0:5:1" \
    --num-frames 100 \
    --max-errors 50
```

**跳过特定测试：**
```bash
# 仅运行BER仿真
python benchmarks/run_benchmark.py \
    --skip-throughput \
    --skip-complexity

# 仅运行吞吐量测试
python benchmarks/run_benchmark.py \
    --skip-ber \
    --skip-complexity
```

**自定义配置文件：**
```bash
python benchmarks/run_benchmark.py \
    --polar-config config/polar_config.yaml \
    --ldpc-config config/ldpc_config.yaml \
    --output-dir my_results
```

**启用第三方库对比：**
```bash
python benchmarks/run_benchmark.py \
    --use-third-party \
    --snr-range "0:5:1" \
    --num-frames 100
```

### 单独运行各模块

每个模块都可以独立运行进行测试：

```bash
# BER仿真
python benchmarks/ber_simulation.py

# 吞吐量测试
python benchmarks/throughput_test.py

# 复杂度分析
python benchmarks/complexity_analysis.py
```

## 命令行选项

```
--snr-range START:STOP:STEP    BER仿真的SNR范围（默认：-2:6:0.5）
--num-frames N                  每个SNR点的最大帧数（默认：1000）
--max-errors N                  达到N个帧错误后停止（默认：100）
--polar-config FILE             Polar配置YAML文件（默认：config/polar_config.yaml）
--ldpc-config FILE              LDPC配置YAML文件（默认：config/ldpc_config.yaml）
--output-dir DIR                输出目录（默认：results）
--skip-ber                      跳过BER仿真
--skip-throughput               跳过吞吐量测试
--skip-complexity               跳过复杂度分析
--use-third-party               启用第三方库对比
--throughput-iterations N       吞吐量测试的迭代次数（默认：100）
```

## 输出结果

结果保存在指定的输出目录（默认：`results/`）：

```
results/
├── figures/
│   ├── ber_curves.png          # BER vs SNR曲线
│   ├── fer_curves.png          # FER vs SNR曲线
│   └── complexity_comparison.png  # 复杂度对比图
└── data/
    ├── ber_simulation_results.json    # BER仿真结果
    ├── throughput_results.json        # 吞吐量测试结果
    ├── complexity_results.json        # 复杂度分析结果
    └── benchmark_results.json         # 综合结果
```

## 配置文件

编辑 `config/polar_config.yaml` 和 `config/ldpc_config.yaml` 来自定义编码参数：

**Polar配置：**
```yaml
encoding:
  N: 1024        # 码长（2的幂次）
  K: 512         # 信息位数

construction:
  method: "ga"   # 构造方法
  design_snr_db: 2.0  # 设计SNR
```

**LDPC配置：**
```yaml
encoding:
  n: 504         # 码长
  k: 252         # 信息位数
  dv: 3          # 变量节点度数
  dc: 6          # 校验节点度数

decoding:
  max_iterations: 50      # 最大迭代次数
  algorithm: "bp"         # 或"ms"表示最小和算法
```

## 示例结果

**典型BER性能（SNR = 2dB）：**
- Polar码：BER ≈ 0.33，FER = 1.0
- LDPC：BER ≈ 0.03，FER = 0.91

**吞吐量（N=128，K=64）：**
- Polar编码：~0.18 Mbps
- Polar解码：~0.19 Mbps
- LDPC编码：~0.01 Mbps
- LDPC解码：~0.00 Mbps（由于BP迭代较慢）

**复杂度（N=128，K=64）：**
- Polar编码：896次异或操作，O(N log N)
- Polar解码：1,024次操作，O(N log N)
- LDPC编码：~3,600次操作，O(m × k)
- LDPC解码：~90,000次操作（50次迭代），O(I × 边数）

## 注意事项

### 性能考虑

- **LDPC解码较慢**：Python中50次迭代的BP解码器处理1000帧需要约20-30秒
  - 为了更快的测试，使用 `--throughput-iterations 50`（默认现在是100）
  - 或使用 `--skip-throughput` 跳过吞吐量测试
  
- **快速测试**：使用快速测试脚本：
  ```bash
  python quick_benchmark.py  # 约3-5分钟，包含第三方库对比
  ```

- **LDPC编码警告**："Could not create systematic generator matrix"对某些参数是正常的
  - 会使用直接求解方法（较慢但正确）
  
- **吞吐量**：Python开销很大；C/C++实现会快10-100倍

- **早停机制**：BER仿真在达到 `--max-errors` 个帧错误后停止，减少高SNR下的测试时间

- **发表质量**：用于研究论文时，使用：
  ```bash
  python benchmarks/run_benchmark.py \
      --snr-range "-2:6:0.5" \
      --num-frames 10000 \
      --max-errors 200 \
      --throughput-iterations 1000
  ```

## 第三方库对比

### 安装

首先，安装所需的第三方库：

```bash
pip install pyldpc py-polar-codes
```

### 使用方法

使用 `--use-third-party` 标志启用第三方库对比：

```bash
python benchmarks/run_benchmark.py --use-third-party
```

这将会：
- 同时运行自实现和第三方库版本
- 生成包含4条曲线的图表：Polar（自实现）、Polar（库）、LDPC（自实现）、LDPC（库）
- 对比性能和正确性

**注意：** 第三方库通常显示更好的性能，因为：
- `polarcodes` 使用正确的软判决SC解码器（而我们的是硬判决）
- `pyldpc` 有使用numba加速的优化BP实现
- 两个库都经过充分测试且可用于生产环境

此对比功能适用于：
- 验证自实现的正确性
- 理解性能差距
- 与最先进技术进行基准测试
