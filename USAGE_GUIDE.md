# 使用指南 (Usage Guide)

快速开始使用PolarCode和LDPC性能测试系统。

## 🚀 快速开始

### 1. 快速开始测试（推荐新手）

```bash
# 运行基础BER/FER测试（5-10分钟）
python benchmarks/run_benchmark.py --snr-range 0:5:1 --num-frames 100 --use-third-party
```

**输出**：
- `results/figures/ber_curves.png` - BER性能对比
- `results/figures/fer_curves.png` - FER性能对比
- `results/figures/complexity_comparison.png` - 复杂度对比
- `results/data/*.json` - 所有结果数据

**特点**：
- ✅ 包含Polar和LDPC自实现与第三方库对比
- ✅ 自动生成BER/FER曲线图
- ✅ 验证算法正确性

---

### 2. 自定义测试

#### 基础BER测试（无第三方库）
```bash
python benchmarks/run_benchmark.py \
    --snr-range "0:5:1" \
    --num-frames 100
```

#### 完整测试（含第三方库对比）
```bash
python benchmarks/run_benchmark.py \
    --snr-range "0:5:1" \
    --num-frames 100 \
    --use-third-party
```

#### 快速测试（跳过慢速部分）
```bash
python benchmarks/run_benchmark.py \
    --snr-range "2:4:1" \
    --num-frames 20 \
    --skip-throughput \
    --use-third-party
```

#### 只测试BER（最快）
```bash
python benchmarks/run_benchmark.py \
    --snr-range "2:4:1" \
    --num-frames 20 \
    --skip-throughput \
    --skip-complexity
```

---

## 📊 运行单个测试模块

### BER仿真
```bash
python benchmarks/ber_simulation.py
```
- 测试Polar和LDPC的BER/FER性能
- 包含第三方库对比（默认启用）
- 生成BER/FER曲线图

### 吞吐量测试
```bash
python benchmarks/throughput_test.py
```
- 测量编码/解码速度（Mbps）
- 注意：LDPC BP解码较慢（~20-30秒/1000帧）

### 复杂度分析
```bash
python benchmarks/complexity_analysis.py
```
- 理论复杂度估算
- 操作数和内存使用
- 生成对比柱状图

---

## ⚙️ 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--snr-range` | SNR范围 (start:stop:step) | "-2:6:0.5" |
| `--num-frames` | 每个SNR点的测试帧数 | 1000 |
| `--max-errors` | 错误帧数阈值（早停） | 100 |
| `--throughput-iterations` | 吞吐量测试迭代次数 | 100 |
| `--use-third-party` | 启用第三方库对比 | 关闭 |
| `--skip-ber` | 跳过BER测试 | - |
| `--skip-throughput` | 跳过吞吐量测试 | - |
| `--skip-complexity` | 跳过复杂度分析 | - |
| `--output-dir` | 输出目录 | "results" |

---

## 💡 使用建议

### 场景1：快速验证功能
```bash
python benchmarks/run_benchmark.py --snr-range 0:5:1 --num-frames 100 --use-third-party
```
**时间**：5-10分钟  
**适用**：初次使用、功能演示、算法验证

### 场景2：详细性能测试
```bash
python benchmarks/run_benchmark.py \
    --snr-range "0:6:0.5" \
    --num-frames 500 \
    --use-third-party
```
**时间**：15-20分钟  
**适用**：性能分析、对比研究

### 场景3：发布质量测试
```bash
python benchmarks/run_benchmark.py \
    --snr-range "-2:8:0.5" \
    --num-frames 10000 \
    --max-errors 200 \
    --throughput-iterations 1000 \
    --use-third-party
```
**时间**：数小时  
**适用**：论文发表、正式报告

### 场景4：仅BER曲线（最快）
```bash
python benchmarks/run_benchmark.py \
    --snr-range "0:5:1" \
    --num-frames 50 \
    --skip-throughput \
    --skip-complexity
```
**时间**：1-2分钟  
**适用**：快速对比、调试代码

---

## 🐛 常见问题

### 1. 程序运行很慢，卡住了？

**原因**：LDPC解码在大量迭代时很慢

**解决方案**：
```bash
# 方案1：减少迭代次数
python benchmarks/run_benchmark.py --throughput-iterations 50

# 方案2：跳过吞吐量测试
python benchmarks/run_benchmark.py --skip-throughput

# 方案3：减少测试帧数
python benchmarks/run_benchmark.py --num-frames 50 --skip-throughput
```

### 2. 出现"Could not create systematic generator matrix"警告？

**原因**：某些LDPC参数下无法生成系统码形式

**解决方案**：这是正常的，程序会自动使用直接求解方法（稍慢但正确）

### 3. 第三方库性能远好于自实现？

**原因**：
- `polarcodes`使用软判决SC解码器（我们用硬判决）
- `pyldpc`使用numba加速的优化BP算法
- 两者都是经过充分测试的生产级代码

**解决方案**：这是正常的，第三方库用于验证正确性和理解优化空间

### 4. 图中看不到第三方库的曲线？

**原因**：第三方库性能太好，BER=0，在对数坐标下无法直接显示

**解决方案**：
- 已自动修复：零值BER显示为1e-6，零值FER显示为1e-4
- 图中会显示为水平虚线（接近图底部）
- 标记样式：方块（□）表示Polar Library，菱形（◇）表示LDPC Library
- 图左下角有注释说明零值的处理方式

### 5. 想要更快的速度？

当前是纯Python实现，较慢。优化方向：
- 使用Numba JIT编译（加速5-10×）
- 使用Cython（加速10-50×）
- 使用C/C++实现（加速50-100×）
- GPU加速（加速100-1000×）

---

## 📁 输出文件说明

运行后，所有结果保存在 `results/` 目录：

```
results/
├── figures/
│   ├── ber_curves.png          # BER vs SNR曲线（含4条线）
│   ├── fer_curves.png          # FER vs SNR曲线（含4条线）
│   └── complexity_comparison.png  # 复杂度对比柱状图
└── data/
    ├── ber_simulation_results.json    # BER测试原始数据
    ├── throughput_results.json        # 吞吐量测试结果
    ├── complexity_results.json        # 复杂度分析结果
    └── benchmark_results.json         # 所有结果汇总
```

---

## 🎯 下一步

1. **查看结果图表**：
   ```bash
   explorer results\figures
   ```

2. **分析JSON数据**：
   ```python
   import json
   with open('results/data/benchmark_results.json') as f:
       data = json.load(f)
   ```

3. **自定义分析**：
   参考 `benchmarks/ber_simulation.py` 编写自己的测试脚本

4. **优化代码**：
   - 改进Polar解码器（当前用硬判决，可改为软判决）
   - 优化LDPC编码器（改进系统码生成）
   - 添加Numba加速

---

## 📚 更多文档

- `benchmarks/README.md` - 详细的benchmark系统说明
- `FIXES_SUMMARY.md` - 最新修复和改进记录
- `README.md` - 项目总体介绍
- `ARCHITECTURE.md` - 项目架构说明

---

**祝使用愉快！** 🎉

如有问题，请查看文档或检查代码注释。
