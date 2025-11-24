# Debug Scripts

这个目录包含了开发过程中使用的调试和分析脚本。这些脚本不是项目的核心功能，但对于理解实现细节和性能调优很有帮助。

## 文件说明

### LDPC相关

- **analyze_ldpc_performance.py** - 分析LDPC BP解码器的性能瓶颈
  - 统计迭代次数分布
  - 测量编码/解码时间
  - 计算理论复杂度
  
- **compare_self_vs_lib.py** - 对比自实现和pyldpc库的性能
  - 相同配置下的速度对比
  - 吞吐量对比

- **test_ldpc_matrix.py** - 测试LDPC H矩阵的生成和性质

### Polar相关

- **check_bit_reversal.py** - 检查Polar编码中的比特反转
  
- **check_lib_decoder_type.py** - 检查第三方库解码器类型
  
- **compare_decoders_same_input.py** - 用相同输入对比不同解码器
  
- **compare_step_by_step.py** - 逐步对比编解码过程
  
- **inspect_polarcodes.py** - 检查polarcodes库的内部实现
  
- **trace_llr_updates.py** - 追踪SC解码器的LLR更新过程
  
- **verify_encoding.py** - 验证Polar编码正确性

### 综合测试

- **quick_benchmark.py** - 快速基准测试脚本
  - 使用较少迭代次数
  - 适合快速验证修改

## 使用方法

这些脚本可以直接从项目根目录运行：

```bash
# 分析LDPC性能
python debug_scripts/analyze_ldpc_performance.py

# 快速基准测试
python debug_scripts/quick_benchmark.py

# 对比自实现和库
python debug_scripts/compare_self_vs_lib.py
```

## 注意事项

⚠️ **这些脚本可能：**
- 使用硬编码的参数
- 依赖特定的项目状态
- 输出大量调试信息
- 不保证向后兼容

**建议仅用于开发和调试，不要用于生产环境。**

如果需要正式的测试和基准测试，请使用 `benchmarks/` 目录下的脚本：
- `benchmarks/run_benchmark.py` - 完整基准测试
- `benchmarks/test_code_parameters.py` - 码长和码率测试  
- `benchmarks/test_snr_curves.py` - SNR性能曲线测试
