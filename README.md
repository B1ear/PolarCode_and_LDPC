# Polar Code & LDPC 编解码器实现与性能对比

本项目完整实现了Polar Code和LDPC两种信道编码方案，并提供性能对比分析工具。

## 项目结构

```
PolarCode_and_LDPC/
├── README.md                      # 项目说明文档
├── ARCHITECTURE.md                # 架构设计文档
├── USAGE_GUIDE.md                 # 使用指南
├── requirements.txt               # Python依赖
├── config/                        # 配置文件
│   ├── polar_config.yaml         # Polar Code配置
│   └── ldpc_config.yaml          # LDPC配置
├── src/                          # 源代码目录
│   ├── polar/                    # Polar Code实现
│   │   ├── encoder.py           # Polar编码器
│   │   ├── decoder.py           # SC/SCL解码器
│   │   ├── construction.py      # 码构造算法
│   │   └── utils.py             # 工具函数
│   ├── ldpc/                     # LDPC实现
│   │   ├── encoder.py           # LDPC编码器
│   │   ├── decoder.py           # BP/MS解码器
│   │   ├── matrix.py            # 校验矩阵生成
│   │   └── utils.py             # 工具函数
│   ├── channel/                  # 信道模拟
│   │   ├── awgn.py              # AWGN信道
│   │   ├── bsc.py               # BSC信道
│   │   └── fading.py            # 衰落信道
│   ├── lib_wrappers/            # 第三方库封装
│   │   ├── polar_wrapper.py     # polarcodes库封装
│   │   └── ldpc_wrapper.py      # pyldpc库封装
│   └── utils/                    # 通用工具
│       ├── metrics.py           # 性能指标计算
│       └── visualization.py     # 可视化工具
├── tests/                        # 单元测试
│   └── test_scl_decoder.py      # SCL解码器测试
├── benchmarks/                   # 性能测试
│   ├── README.md                # 测试系统说明
│   ├── run_benchmark.py         # 主测试脚本
│   ├── ber_simulation.py        # BER/FER仿真
│   ├── throughput_test.py       # 吞吐量测试
│   ├── complexity_analysis.py   # 复杂度分析
│   ├── test_code_parameters.py  # 码长码率测试
│   ├── test_snr_curves.py       # SNR性能曲线
│   └── benchmark_scl.py         # SCL解码器性能测试
├── docs/                         # 详细文档
│   ├── SCL_DECODER_README.md    # SCL解码器文档
│   ├── SNR_CURVES_TEST_SUMMARY.md  # SNR测试报告
│   └── ...                      # 其他技术文档
└── results/                      # 测试结果
    ├── figures/                 # BER/FER曲线图
    ├── data/                    # 原始数据(JSON)
    ├── code_params/             # 码长码率测试结果
    └── snr_curves/              # SNR曲线测试结果
```

## 最新更新

### v2.0 - SCL解码器实现
- ✅ 实现完整的SCL (Successive Cancellation List) 解码器
- ✅ 支持可配置的列表大小，提升解码性能
- ✅ 添加SCL性能测试和对比工具
- ✅ 优化项目结构，移除冗余代码
- ✅ 完善文档和使用示例

## 功能特性

### Polar Code
- **编码器**: 支持任意码长N(2的幂次)和信息长度K
  - Kronecker乘积高效编码
  - 支持CRC附加（用于CA-SCL）
- **SC解码器**: 
  - 连续消除(Successive Cancellation)解码
  - 软判决LLR输入
  - 与polarcodes库性能一致
- **SCL解码器**: 
  - 连续消除列表(Successive Cancellation List)解码
  - 维护多个候选路径，提高解码性能
  - 支持可配置的列表大小(L)
  - 可选CRC辅助路径选择
  - 详见 [SCL解码器文档](docs/SCL_DECODER_README.md)
- **码构造**: 
  - 使用polarcodes库预计算冻结位集合
  - 基于Bhattacharyya参数的最优构造
  - 确保有限码长下的构造质量

### LDPC
- **编码器**: ✅ 已实现并优化
  - 基于校验矩阵H和生成矩阵G
  - 支持规则LDPC码 (dv, dc可配置)
  - 向量化GF(2)运算，编码速度快
- **BP解码器**: ✅ 已实现并优化
  - Belief Propagation迭代解码
  - 预构建索引映射表（3.8倍加速）
  - 早停机制（所有校验通过即停止）
  - max_iter=20，在性能和速度间平衡
- **MS解码器**: ✅ 已实现
  - Min-Sum算法（BP的简化版）
  - 更快的解码速度
- **矩阵构造**:
  - 使用pyldpc生成H和G矩阵
  - MacKay构造方法

### 性能评估 ✅ 已完成

#### 已完成的测试
1. **基础BER/FER测试** (`results/figures/`)
   - SNR: 0-5 dB，典型码长N≈1000
   - 自实现与第三方库对比

2. **码长与码率测试** (`results/code_params/`)
   - 6种码长 × 10种码率
   - SNR=3dB固定条件
   - 复杂度扩展性分析

3. **SNR性能曲线** (`results/snr_curves/`) ⭐ 最重要
   - 4种码率: [0.50, 0.67, 0.75, 0.83]
   - SNR: -2 to 5 dB (步长1dB)
   - 100帧/SNR点，最多100错误帧
   - 量化SNR门限与编码增益

#### 核心发现
- **低码率(≤0.5)**: Polar ≈ LDPC
- **中码率(0.67)**: LDPC优2 dB
- **高码率(≥0.75)**: LDPC优3-4 dB，Polar性能劣化
- 验证了5G NR标准设计的合理性

### 验证与对比 ✅
- 与polarcodes和pyldpc库高度一致
- 自实现BER/FER曲线与库几乎重合
- 通过修复关键bug确保正确性

## 安装

```bash
# 克隆或进入项目目录
cd PolarCode_and_LDPC

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 基本使用

```python
from src.polar import PolarEncoder, SCDecoder, SCLDecoder
from src.ldpc import LDPCEncoder, BPDecoder
from src.channel import AWGNChannel
import numpy as np

# Polar Code示例 - SC解码器
from src.lib_wrappers import PolarLibWrapper

# 使用库生成冻结位集合（最优构造）
N, K = 256, 128
lib = PolarLibWrapper(N, K, design_snr_db=2.0)
frozen_bits = lib.get_frozen_bits_positions()

polar_enc = PolarEncoder(N, K, frozen_bits=frozen_bits)
polar_dec = SCDecoder(N, K, frozen_bits=frozen_bits)

message = np.random.randint(0, 2, K)
codeword = polar_enc.encode(message)

# 通过AWGN信道
channel = AWGNChannel(snr_db=3.0)
llr = channel.transmit(codeword, return_llr=True)

# SC解码（使用LLR软判决）
decoded = polar_dec.decode(llr)

# Polar Code示例 - SCL解码器（更好的性能）
scl_dec = SCLDecoder(N, K, list_size=8, frozen_bits=frozen_bits)
decoded_scl = scl_dec.decode(llr)

# LDPC示例
from src.lib_wrappers import LDPCLibWrapper

# 使用库生成H和G矩阵（公平对比）
lib = LDPCLibWrapper(n=504, k=252, dv=3, dc=6, seed=42)
H = lib.get_parity_check_matrix()
G = lib.get_generator_matrix()

ldpc_enc = LDPCEncoder(n=504, k=lib.k, H=H, G=G)
ldpc_dec = BPDecoder(H, max_iter=20)  # 已优化：max_iter=20

message = np.random.randint(0, 2, lib.k)
codeword = ldpc_enc.encode(message)
llr = channel.transmit(codeword, return_llr=True)
decoded_full = ldpc_dec.decode(llr)
decoded = decoded_full[:lib.k]  # 取前k位信息位
```

### 运行性能测试

```bash
# 1. 基础BER/FER测试
python benchmarks/run_benchmark.py \
    --snr-range 0:5:1 \
    --num-frames 100 \
    --use-third-party

# 2. 码长与码率测试
python benchmarks/test_code_parameters.py

# 3. SNR性能曲线（最重要）
python benchmarks/test_snr_curves.py

# 4. 单独的吞吐量和复杂度测试
python benchmarks/throughput_test.py
python benchmarks/complexity_analysis.py

# 5. SCL解码器性能测试
python benchmarks/benchmark_scl.py
```

### 运行测试

```bash
# 运行SCL解码器测试
python tests/test_scl_decoder.py

# 或使用pytest
pytest tests/test_scl_decoder.py -v
```

## 配置

配置文件位于 `config/` 目录，可自定义编码参数、解码算法选项等。

## 结果

性能测试结果将保存在 `results/` 目录：
- `figures/`: BER曲线、性能对比图表
- `data/`: 原始测试数据(CSV/JSON格式)

## 依赖库

- **核心**: numpy, scipy
- **可视化**: matplotlib, seaborn
- **验证**: pyldpc, polarcodes
- **测试**: pytest
- **配置**: pyyaml
- **数据处理**: pandas

## 性能指标

本项目评估以下性能指标：

1. **误码性能**
   - BER vs SNR
   - FER vs SNR
   
2. **计算性能**
   - 编码/解码吞吐量 (Mbps)
   - 每比特计算复杂度
   - 平均解码迭代次数

3. **存储开销**
   - 码表大小

## 📚 详细文档

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - 架构设计文档
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - 使用指南和常见问题
- **[benchmarks/README.md](benchmarks/README.md)** - 性能测试系统说明
- **[docs/](docs/)** - 测试报告和技术文档
  - [SCL解码器实现](docs/SCL_DECODER_README.md)
  - [SNR性能曲线测试](docs/SNR_CURVES_TEST_SUMMARY.md)

## 许可

MIT License

## 参考文献

1. Arikan, E. (2009). "Channel Polarization: A Method for Constructing Capacity-Achieving Codes"
2. Gallager, R.G. (1962). "Low-Density Parity-Check Codes"
3. MacKay, D.J.C. (1999). "Good Error-Correcting Codes Based on Very Sparse Matrices"
