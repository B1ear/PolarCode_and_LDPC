# 项目文档索引

本目录包含项目开发过程中的详细文档、测试报告和技术说明。

## 📊 测试报告

### [SNR_CURVES_TEST_SUMMARY.md](SNR_CURVES_TEST_SUMMARY.md)
**最重要** - SNR性能曲线完整测试报告
- 测试日期: 2025-11-24
- 内容: Polar和LDPC在不同码率下的SNR性能对比
- 核心发现: 
  - 低码率(≤0.5): 两者性能相当
  - 中高码率(0.67-0.83): LDPC优势明显(2-4 dB增益)
  - 理论验证: Polar高码率劣化符合有限码长理论
- 结果: 8个SNR点 × 4个码率 × 2种码 × 2种实现

### [CODE_PARAMS_TEST_SUMMARY_V2.md](CODE_PARAMS_TEST_SUMMARY_V2.md)
码长和码率参数测试报告
- 测试内容: 6种码长 × 10种码率
- 关键发现:
  - 码长扩展性: Polar O(N log N), LDPC O(N)理论复杂度
  - 码率依赖: Polar高码率性能下降，LDPC保持稳定
  - 吞吐量: LDPC编码快12.5倍，解码慢5.2倍

## 🔧 技术文档

### [LDPC_OPTIMIZATION.md](LDPC_OPTIMIZATION.md)
LDPC解码器效率优化详细报告
- 问题: 初始解码慢54倍
- 优化方案: 预构建索引映射表 + 向量化GF(2)求解
- 效果: 3.8倍解码加速，3.2倍编码加速
- 进一步方向: Min-Sum算法、Numba加速、并行化

### [SCL_IMPLEMENTATION_NOTE.md](SCL_IMPLEMENTATION_NOTE.md)
Polar SCL解码器实现笔记
- SCL算法原理
- 实现挑战
- 与SC解码器对比


## 📚 文档使用建议

### 快速了解项目
1. 阅读根目录的 [README.md](../README.md)
2. 查看 [ARCHITECTURE.md](../ARCHITECTURE.md) 了解架构设计
3. 参考 [USAGE_GUIDE.md](../USAGE_GUIDE.md) 快速上手

### 深入理解性能
1. **SNR性能**: [SNR_CURVES_TEST_SUMMARY.md](SNR_CURVES_TEST_SUMMARY.md) - 最重要
2. **参数对比**: [CODE_PARAMS_TEST_SUMMARY_V2.md](CODE_PARAMS_TEST_SUMMARY_V2.md)
3. **优化过程**: [LDPC_OPTIMIZATION.md](LDPC_OPTIMIZATION.md)
4. **SCL笔记**: [SCL_IMPLEMENTATION_NOTE.md](SCL_IMPLEMENTATION_NOTE.md)

## 📝 文档维护

- 测试报告: 每次重大测试后更新
- 技术文档: 实现新功能或优化时更新
- 问题记录: 修复重要bug时更新

最后更新: 2025-11-24
