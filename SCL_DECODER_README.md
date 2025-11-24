# SCL (Successive Cancellation List) Decoder

## 概述

SCL解码器是Polar码的一种改进解码算法，通过维护多个候选解码路径来提高解码性能。与基本的SC (Successive Cancellation) 解码器相比，SCL解码器可以更好地处理信道噪声。

## 实现特点

### 核心机制

1. **路径分裂 (Path Splitting)**
   - 在解码信息位时，每个活跃路径分裂为两个候选（bit=0 和 bit=1）
   - 保留路径度量（对数似然比）最大的L个路径
   - 冻结位不分裂，所有路径强制为0

2. **路径度量 (Path Metrics)**
   - 使用对数似然比 log P(bit|LLR) 作为路径度量
   - 数值稳定的计算方法，避免下溢和上溢
   - 选择度量最大的路径作为最终输出

3. **独立路径状态**
   - 每个路径维护独立的LLR矩阵和比特矩阵
   - 支持高效的路径复制和更新

## 使用方法

### 基本用法

```python
from src.polar.encoder import PolarEncoder
from src.polar.decoder import SCLDecoder
from src.channel.awgn import AWGNChannel
import numpy as np

# 参数设置
N, K = 128, 64  # 码长和信息长度
L = 8           # 列表大小

# 创建编码器和解码器
encoder = PolarEncoder(N, K)
decoder = SCLDecoder(N, K, list_size=L)

# 编码
message = np.random.randint(0, 2, K)
codeword = encoder.encode(message)

# 通过信道
channel = AWGNChannel(snr_db=2.0)
llr = channel.transmit(codeword, return_llr=True)

# 解码
decoded = decoder.decode(llr)

# 验证
print(f"Bit errors: {np.sum(decoded != message)}")
```

### 参数说明

- `N`: 码长（必须是2的幂）
- `K`: 信息位长度
- `list_size` (L): 保留的路径数量
  - L=1: 等价于SC解码器
  - L越大，性能越好，但计算复杂度越高
  - 常用值: 2, 4, 8, 16, 32
- `frozen_bits`: 可选，指定冻结位位置
- `use_crc`: 是否使用CRC辅助路径选择（待实现）
- `crc_polynomial`: CRC多项式类型（待实现）

### 与SC解码器对比

```python
from src.polar.decoder import SCDecoder, SCLDecoder

# SC解码器
sc_decoder = SCDecoder(N, K)
decoded_sc = sc_decoder.decode(llr)

# SCL解码器
scl_decoder = SCLDecoder(N, K, list_size=8)
decoded_scl = scl_decoder.decode(llr)

# 在低SNR下，SCL通常优于SC
```

## 性能基准

运行性能测试：

```bash
# 基本测试
python test_scl_decoder.py

# 完整性能基准
python benchmark_scl.py
```

### 预期性能

- **计算复杂度**: O(L · N · log N)
  - L是列表大小
  - N是码长
  - 相比SC解码器增加了L倍的计算量

- **内存复杂度**: O(L · N · log N)
  - 需要存储L个路径的LLR和比特矩阵

- **解码性能**:
  - L=1: 与SC解码器相同
  - L增加时，帧错误率降低
  - 通常L=8-32可以获得显著的性能提升

## 测试结果示例

```
=== Basic SCL Decoder Test ===

Original message: [0 1 0 0 0 1 0 0]
Codeword: [0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0]

SNR: 2.0 dB

--- Decoding Results ---
L=1: decoded=[0 1 0 0 0 1 0 0], errors=0
L=2: decoded=[0 1 0 0 0 1 0 0], errors=0
L=4: decoded=[0 1 0 0 0 1 0 0], errors=0
L=8: decoded=[0 1 0 0 0 1 0 0], errors=0
```

## 技术细节

### 对数似然比计算

给定 LLR = log(P(bit=0)/P(bit=1))，我们计算：

```
P(bit=0|LLR) = 1 / (1 + e^-LLR)
P(bit=1|LLR) = 1 / (1 + e^LLR)

log P(bit=0|LLR) = -log(1 + e^-LLR)
log P(bit=1|LLR) = -log(1 + e^LLR)
```

使用数值稳定的实现避免上溢/下溢。

### LLR传播

- **上分支 (f函数)**: min-sum近似
  ```
  LLR_out = sign(LLR_1) * sign(LLR_2) * min(|LLR_1|, |LLR_2|)
  ```

- **下分支 (g函数)**:
  ```
  LLR_out = LLR_2 + (1 - 2*bit_1) * LLR_1
  ```

### 路径选择策略

在每个信息位：
1. 计算所有可能的候选路径（当前活跃路径数 × 2）
2. 根据路径度量排序
3. 保留前L个路径
4. 更新活跃路径状态

## 未来改进

1. **CRC辅助SCL (CA-SCL)**
   - 使用CRC校验辅助路径选择
   - 提高短码性能

2. **自适应SCL**
   - 根据SNR动态调整L
   - 平衡性能和复杂度

3. **并行化**
   - 利用GPU加速路径计算
   - 提高吞吐量

4. **混合解码**
   - 结合SC和SCL的优势
   - 自适应选择解码策略

## 参考文献

1. I. Tal and A. Vardy, "List Decoding of Polar Codes," IEEE Trans. Inf. Theory, 2015
2. K. Niu and K. Chen, "CRC-Aided Decoding of Polar Codes," IEEE Commun. Lett., 2012
3. E. Arikan, "Channel Polarization: A Method for Constructing Capacity-Achieving Codes," IEEE Trans. Inf. Theory, 2009

## 贡献者

实现日期: 2025-11-24
分支: feature/scl-decoder
