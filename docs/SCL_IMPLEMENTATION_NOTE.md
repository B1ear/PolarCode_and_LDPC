# SCL解码器实现说明

## 当前状态

SCL (Successive Cancellation List) 解码器的实现遇到了技术挑战。Polar码的SC和SCL解码器需要正确处理复杂的树状解码路径，实现难度较高。

## 问题分析

1. **SC解码器基础问题**: 当前SC解码器在无噪声条件下都无法正确解码，说明基础的LLR计算和比特更新逻辑有问题

2. **实现复杂度**: Polar码解码涉及：
   - 蝶形网络结构的LLR传播
   - f函数和g函数的正确应用
   - 比特值在解码树中的正确更新
   - SCL需要额外的路径管理和度量计算

3. **测试结果**: SCL解码器当前BER约为0.5（接近随机），远差于预期

## 建议方案

### 方案1: 使用成熟的第三方库

推荐使用以下Python库作为参考或直接集成：

1. **polar-codes** (推荐)
   ```bash
   pip install polar-codes
   ```
   - 提供完整的Polar码编解码实现
   - 包含SC, SCL, CA-SCL解码器
   - 有良好的文档和示例

2. **py-polar-codes**
   ```bash
   pip install py-polar-codes
   ```
   - 另一个Polar码实现
   - 包含多种解码算法

3. **commpy**
   ```bash
   pip install scikit-commpy
   ```
   - 综合通信库，包含Polar码

### 方案2: 参考标准实现

参考以下开源项目的实现：
- [polar-3gpp-matlab](https://github.com/robmaunder/polar-3gpp-matlab)
- [PolarCodes](https://github.com/tavildar/Polar)
- Arikan's original paper实现

### 方案3: 简化项目范围

如果主要目的是对比Polar和LDPC：
1. 保留当前的Polar编码器（工作正常）
2. 使用基本的硬判决解码器（虽然性能差但简单）
3. 重点展示LDPC的优势
4. 在文档中说明Polar SCL解码器的实现挑战

## 当前项目完成度

✓ Polar编码器 - 完成且正确  
✓ LDPC编解码器 - 完成且正确  
✓ 信道模型 - 完成  
✓ 性能评估框架 - 完成  
✗ Polar SC/SCL解码器 - 需要修复  

## 下一步行动

建议：
1. **短期**: 集成`polar-codes`库作为Polar解码的参考实现
2. **中期**: 仔细研究标准实现，修复当前SC解码器
3. **长期**: 实现完整的SCL解码器with CRC

## 性能对比仍然有效

即使Polar解码器有问题，当前的LDPC实现是正确的，仍然可以：
- 展示LDPC的编解码过程
- 展示LDPC的性能（BER/FER曲线）
- 与理论极限对比
- 测试不同的LDPC参数

## 参考资料

1. Arikan, E. "Channel Polarization: A Method for Constructing Capacity-Achieving Codes for Symmetric Binary-Input Memoryless Channels" (2009)
2. Tal, I., and Vardy, A. "List Decoding of Polar Codes" (2015)
3. 3GPP TS 38.212 - Polar码标准规范
