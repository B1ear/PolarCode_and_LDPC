# 项目架构详细说明

## 整体架构

本项目采用模块化设计，分为以下几个主要层次：

### 1. 核心编解码层 (src/)

#### 1.1 Polar Code 模块 (src/polar/)
```
polar/
├── encoder.py          # Polar编码器实现
├── decoder.py          # SC/SCL/CA-SCL解码器
├── construction.py     # 信道极化码构造
└── utils.py           # Polar相关工具函数
```

**功能说明：**
- **encoder.py**: 实现Polar码编码器
  - `PolarEncoder` 类：使用Kronecker乘积实现高效编码
  - 支持任意N=2^n的码长
  - CRC附加功能（用于CA-SCL）

- **decoder.py**: 实现多种解码算法
  - `SCDecoder`: 连续消除(Successive Cancellation)解码
  - `SCLDecoder`: 连续消除列表(Successive Cancellation List)解码
  - `CASCLDecoder`: CRC辅助的SCL解码
  - 支持软判决输入(LLR)

- **construction.py**: 码构造算法
  - Bhattacharyya参数计算
  - 高斯近似方法
  - Monte Carlo仿真方法
  - 信息位和冻结位选择

- **utils.py**: 辅助工具
  - 冻结位生成
  - CRC编解码
  - 位反转排列

#### 1.2 LDPC 模块 (src/ldpc/)
```
ldpc/
├── encoder.py          # LDPC编码器
├── decoder.py          # BP/MS解码器
├── matrix.py          # 校验矩阵构造
└── utils.py           # LDPC工具函数
```

**功能说明：**
- **encoder.py**: LDPC编码器
  - `LDPCEncoder` 类：基于生成矩阵或校验矩阵编码
  - 支持规则和非规则LDPC码
  - 系统码实现

- **decoder.py**: 迭代解码算法
  - `BPDecoder`: 置信传播(Belief Propagation)解码
  - `MSDecoder`: 最小和(Min-Sum)解码
  - `NMSDecoder`: 归一化最小和解码
  - `OMSDecoder`: 偏移最小和解码

- **matrix.py**: 校验矩阵生成
  - MacKay构造方法
  - PEG (Progressive Edge Growth) 算法
  - QC-LDPC (Quasi-Cyclic) 构造
  - 规则/非规则LDPC矩阵生成

- **utils.py**: 辅助功能
  - Tanner图创建和可视化
  - 校验和计算
  - 稀疏矩阵优化

#### 1.3 信道模拟层 (src/channel/)
```
channel/
├── awgn.py            # AWGN信道
├── bsc.py             # 二进制对称信道
└── fading.py          # 衰落信道
```

**功能说明：**
- **awgn.py**: 加性高斯白噪声信道
  - BPSK调制
  - SNR到噪声方差转换
  - LLR计算

- **bsc.py**: 二进制对称信道
  - 交叉概率控制
  - 硬判决输入

- **fading.py**: 衰落信道模拟
  - Rayleigh衰落
  - Rician衰落

### 2. 验证和对比层 (src/lib_wrappers/)

```
lib_wrappers/
├── polar_wrapper.py    # Polar库封装
└── ldpc_wrapper.py     # LDPC库封装
```

**功能说明：**
- 封装第三方库（pyldpc, commpy等）
- 提供统一接口用于正确性验证
- 性能对比基准

### 3. 工具层 (src/utils/)

```
utils/
├── metrics.py          # 性能指标计算
└── visualization.py    # 结果可视化
```

**功能说明：**
- **metrics.py**: 
  - BER (Bit Error Rate) 计算
  - FER (Frame Error Rate) 计算
  - 吞吐量统计
  - 计算复杂度度量

- **visualization.py**:
  - BER vs SNR曲线绘制
  - 性能对比图表
  - Tanner图可视化
  - 结果导出（PNG/PDF/SVG）

### 4. 测试层 (tests/)

```
tests/
├── test_polar.py           # Polar Code单元测试
├── test_ldpc.py            # LDPC单元测试
└── test_correctness.py     # 正确性验证测试
```

**测试策略：**
- 单元测试：测试各模块独立功能
- 集成测试：测试编解码流程
- 正确性验证：与第三方库对比
- 边界条件测试：极端参数测试

### 5. 性能测试层 (benchmarks/)

```
benchmarks/
├── run_benchmark.py        # 主测试脚本
├── ber_simulation.py       # BER仿真
├── throughput_test.py      # 吞吐量测试
└── complexity_analysis.py  # 复杂度分析
```

**性能测试内容：**
- BER/FER vs SNR仿真
- 编解码吞吐量测试
- 时间复杂度分析
- 空间复杂度分析
- 对比分析报告生成

## 数据流

```
输入数据 (信息位)
    ↓
编码器 (Polar/LDPC)
    ↓
调制 (BPSK)
    ↓
信道 (AWGN/BSC/Fading)
    ↓
解调 (软/硬判决)
    ↓
解码器 (SC/SCL/BP/MS)
    ↓
输出数据 (估计信息位)
    ↓
性能评估 (BER/FER计算)
```

## 关键算法实现

### Polar Code

1. **编码过程**:
   - 输入: u (长度N, 包含K个信息位和N-K个冻结位)
   - 过程: x = u * G_N, 其中 G_N = F^⊗n, F = [[1,0],[1,1]]
   - 输出: x (长度N的码字)

2. **SC解码**:
   - 从i=0到N-1依次解码每个位
   - 使用LLR递归计算
   - 冻结位已知，信息位通过硬判决获得

3. **SCL解码**:
   - 维护L个候选路径
   - 每个位置分裂为2L个路径
   - 选择度量最好的L个路径
   - CRC辅助选择最终路径

### LDPC

1. **编码过程**:
   - 输入: u (长度k的信息位)
   - 过程: x = u * G 或求解 H * x^T = 0
   - 输出: x (长度n的码字)

2. **BP解码**:
   - 初始化变量节点消息
   - 迭代更新校验节点和变量节点消息
   - 消息传递直到收敛或达到最大迭代次数
   - 硬判决输出

3. **Min-Sum解码**:
   - BP算法的简化版本
   - 使用最小值代替tanh/log运算
   - 归一化或偏移修正提高性能

## 配置系统

使用YAML配置文件管理参数：
- `config/polar_config.yaml`: Polar Code配置
- `config/ldpc_config.yaml`: LDPC配置

可配置内容：
- 编码参数 (N, K, n, k, rate)
- 解码算法选择
- 性能测试参数
- 可视化选项

## 扩展性设计

### 添加新的编码方案
1. 在`src/`下创建新模块
2. 实现encoder和decoder类
3. 遵循现有接口约定
4. 添加对应测试和benchmark

### 添加新的信道类型
1. 在`src/channel/`添加新文件
2. 继承基础信道类
3. 实现transmit方法

### 添加新的解码算法
1. 在对应的decoder.py中添加新类
2. 实现decode方法
3. 添加配置支持

## 性能优化策略

1. **NumPy向量化**: 避免Python循环
2. **稀疏矩阵**: 使用scipy.sparse存储LDPC矩阵
3. **LLR域计算**: 避免概率域乘法
4. **提前终止**: CRC检查或校验和满足时停止
5. **并行化**: 多进程处理不同SNR点

## 代码风格

- 遵循PEP 8
- 使用类型提示（Type Hints）
- 详细的docstring（Google风格）
- 单元测试覆盖率 > 80%

## 依赖管理

核心依赖：
- numpy: 数值计算
- scipy: 科学计算和稀疏矩阵
- matplotlib/seaborn: 可视化
- pyyaml: 配置管理
- pytest: 测试框架

验证依赖：
- pyldpc: LDPC参考实现
- commpy: 通信系统库

## 输出结果

结果保存在`results/`目录：
- `figures/`: 图表文件
- `data/`: 原始数据（JSON/CSV）

数据格式：
```json
{
  "polar": {
    "ber": [list],
    "fer": [list],
    "snr_db": [list],
    "throughput_mbps": float
  },
  "ldpc": {
    "ber": [list],
    "fer": [list],
    "snr_db": [list],
    "throughput_mbps": float
  }
}
```

## 开发路线图

### Phase 1: 基础实现 (当前)
- [x] 项目架构设计
- [ ] Polar Code编解码器
- [ ] LDPC编解码器
- [ ] AWGN信道模拟
- [ ] 基础性能测试

### Phase 2: 高级功能
- [ ] SCL解码优化
- [ ] BP解码优化
- [ ] 更多信道类型
- [ ] 第三方库集成

### Phase 3: 性能优化
- [ ] C/C++扩展
- [ ] GPU加速
- [ ] 并行化处理
- [ ] 内存优化

### Phase 4: 分析工具
- [ ] 交互式Jupyter笔记本
- [ ] 实时性能监控
- [ ] 详细的复杂度分析
- [ ] Web界面（可选）
