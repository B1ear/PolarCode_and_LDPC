"""
SC vs SCL Decoder Performance Comparison and Visualization
统一的SC vs SCL对比工具 - 支持快速演示和完整对比两种模式
"""

import numpy as np
import sys
from pathlib import Path
import time
import json

# 设置matplotlib为非交互式后端（必须在导入pyplot之前）
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，不需要显示窗口
import matplotlib.pyplot as plt

from typing import Dict, List
import argparse

# 设置字体为英文
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polar import PolarEncoder, SCDecoder, SCLDecoder
from channel import AWGNChannel
from lib_wrappers import PolarLibWrapper


# ============================================================================
# 快速演示模式 (Quick Demo Mode)
# ============================================================================

def quick_demo(N: int = 64, K: int = 32, snr_db: float = 1.5, 
               n_trials: int = 500):
    """
    快速演示模式 - 单个SNR点的性能对比
    
    Args:
        N: 码长
        K: 信息位长度
        snr_db: 信噪比 (dB)
        n_trials: 试验次数
    """
    
    print("="*70)
    print("SC vs SCL Quick Demo (快速演示模式)")
    print("="*70)
    
    print(f"\n参数配置:")
    print(f"  码长 (N):        {N}")
    print(f"  信息位长 (K):    {K}")
    print(f"  编码率:          {K/N:.3f}")
    print(f"  信噪比 (SNR):    {snr_db} dB")
    print(f"  试验次数:        {n_trials}")
    
    # 初始化
    print(f"\n初始化编码器和解码器...")
    lib = PolarLibWrapper(N, K, 2.0)
    frozen_bits = lib.get_frozen_bits_positions()
    encoder = PolarEncoder(N, K, frozen_bits=frozen_bits)
    
    sc_decoder = SCDecoder(N, K, frozen_bits=frozen_bits)
    scl_decoders = {
        L: SCLDecoder(N, K, list_size=L, frozen_bits=frozen_bits)
        for L in [1, 2, 4, 8]
    }
    
    channel = AWGNChannel(snr_db)
    
    # 统计
    sc_errors = 0
    sc_decode_time = 0
    scl_errors = {L: 0 for L in scl_decoders.keys()}
    scl_decode_time = {L: 0 for L in scl_decoders.keys()}
    
    np.random.seed(42)
    
    print(f"\n运行仿真...")
    for trial in range(n_trials):
        msg = np.random.randint(0, 2, K)
        cw = encoder.encode(msg)
        llr = channel.transmit(cw, return_llr=True)
        
        # SC解码
        start_time = time.time()
        decoded_sc = sc_decoder.decode(llr.copy())
        sc_decode_time += time.time() - start_time
        if not np.array_equal(decoded_sc, msg):
            sc_errors += 1
        
        # SCL解码
        for L, decoder in scl_decoders.items():
            start_time = time.time()
            decoded_scl = decoder.decode(llr.copy())
            scl_decode_time[L] += time.time() - start_time
            if not np.array_equal(decoded_scl, msg):
                scl_errors[L] += 1
        
        if (trial + 1) % 100 == 0:
            print(f"  已完成 {trial + 1}/{n_trials}")
    
    # 计算性能指标
    sc_ber = sc_errors / n_trials
    sc_time_per_frame = sc_decode_time / n_trials * 1000  # ms
    
    # 结果输出
    print(f"\n{'='*70}")
    print("演示结果:")
    print(f"{'='*70}")
    print(f"SC (L=1):     {sc_errors:4d}/{n_trials} 错误 ({sc_ber*100:6.2f}%) | "
          f"时间: {sc_time_per_frame:.3f} ms/frame")
    
    for L in sorted(scl_decoders.keys()):
        errors = scl_errors[L]
        ber = errors / n_trials
        time_per_frame = scl_decode_time[L] / n_trials * 1000
        improvement = (sc_ber - ber) / sc_ber * 100 if sc_ber > 0 else 0
        print(f"SCL (L={L}):   {errors:4d}/{n_trials} 错误 ({ber*100:6.2f}%) | "
              f"时间: {time_per_frame:.3f} ms/frame | 改进: {improvement:5.1f}%")
    
    # 简单可视化
    print(f"\n生成可视化...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 柱状图1: 错误数
    labels = ['SC(L=1)'] + [f'SCL(L={L})' for L in sorted(scl_decoders.keys())]
    errors_list = [sc_errors] + [scl_errors[L] for L in sorted(scl_decoders.keys())]
    colors = ['#FF6B6B'] + ['#51CF66', '#339AF0', '#FFD43B', '#DA77F2']
    
    ax1.bar(labels, errors_list, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Error Count', fontsize=11, fontweight='bold')
    ax1.set_title(f'Error Count Comparison (N={N}, K={K}, SNR={snr_db}dB)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(errors_list):
        ax1.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 柱状图2: 错误率
    error_rates = [e/n_trials*100 for e in errors_list]
    ax2.bar(labels, error_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Error Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Error Rate Comparison', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(error_rates):
        ax2.text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 柱状图3: 解码时间
    times_list = [sc_time_per_frame] + [scl_decode_time[L]/n_trials*1000 
                                        for L in sorted(scl_decoders.keys())]
    ax3.bar(labels, times_list, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Decoding Time (ms/frame)', fontsize=11, fontweight='bold')
    ax3.set_title('Decoding Time Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(times_list):
        ax3.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 柱状图4: 性能改进百分比
    improvements = [0]  # SC作为基准
    for L in sorted(scl_decoders.keys()):
        scl_ber = scl_errors[L] / n_trials
        improvement = (sc_ber - scl_ber) / sc_ber * 100 if sc_ber > 0 else 0
        improvements.append(max(0, improvement))
    
    colors_imp = ['gray'] + colors[1:]
    ax4.bar(labels, improvements, color=colors_imp, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Performance Improvement (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Improvement vs SC', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 105])
    for i, v in enumerate(improvements):
        ax4.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # 修复保存路径 - 使用绝对路径
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / 'results' / 'sc_scl_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'sc_scl_quick_demo.png'
    
    try:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ 可视化已保存: {output_file.absolute()}")
    except Exception as e:
        print(f"✗ 保存图片失败: {e}")
        print(f"  尝试的路径: {output_file.absolute()}")
    finally:
        plt.close()
    
    print(f"\n{'='*70}")
    print("快速演示完成!")
    print(f"{'='*70}\n")


# ============================================================================
# 完整对比模式 (Full Comparison Mode)
# ============================================================================

def simulate_sc_vs_scl(
    N: int,
    K: int,
    snr_range: np.ndarray,
    list_sizes: List[int] = [1, 2, 4, 8, 16],
    num_frames: int = 100,
    max_errors: int = 100,
    seed: int = 42
) -> Dict:
    """
    SC与SCL解码性能对比仿真
    
    Args:
        N: 码长
        K: 信息位长度
        snr_range: SNR范围 (dB)
        list_sizes: SCL列表大小列表
        num_frames: 每个SNR点的最大帧数
        max_errors: 达到此错误数后提前停止
        seed: 随机种子
        
    Returns:
        包含BER/FER结果的字典
    """
    print(f"\n{'='*70}")
    print(f"SC vs SCL 完整对比: N={N}, K={K}, 编码率={K/N:.3f}")
    print(f"{'='*70}")
    print(f"SNR范围:          {snr_range[0]:.1f} 到 {snr_range[-1]:.1f} dB")
    print(f"列表大小:         {list_sizes}")
    print(f"每SNR点最大帧数:  {num_frames}")
    print(f"最大错误数:       {max_errors}")
    
    # 初始化编码器和解码器
    print(f"\n初始化编码器和解码器...")
    lib = PolarLibWrapper(N, K, design_snr_db=2.0)
    frozen_bits = lib.get_frozen_bits_positions()
    encoder = PolarEncoder(N, K, frozen_bits=frozen_bits)
    
    # SC解码器
    sc_decoder = SCDecoder(N, K, frozen_bits=frozen_bits)
    
    # SCL解码器
    scl_decoders = {
        L: SCLDecoder(N, K, list_size=L, frozen_bits=frozen_bits)
        for L in list_sizes
    }
    
    results = {
        'N': N,
        'K': K,
        'rate': K/N,
        'snr_db': snr_range.tolist(),
        'sc': {
            'ber': [],
            'fer': [],
            'time': []
        },
        'scl': {L: {'ber': [], 'fer': [], 'time': []} for L in list_sizes}
    }
    
    # 仿真
    np.random.seed(seed)
    
    for snr_db in snr_range:
        print(f"\nSNR = {snr_db} dB", end='')
        channel = AWGNChannel(snr_db)
        
        # SC解码统计
        sc_bit_errors = 0
        sc_frame_errors = 0
        sc_total_bits = 0
        sc_decode_time = 0
        
        # SCL解码统计
        scl_bit_errors = {L: 0 for L in list_sizes}
        scl_frame_errors = {L: 0 for L in list_sizes}
        scl_decode_time = {L: 0 for L in list_sizes}
        scl_total_bits = {L: 0 for L in list_sizes}
        
        frame_count = 0
        
        # 遍历帧
        for frame_idx in range(num_frames):
            # 生成消息
            message = np.random.randint(0, 2, K)
            
            # 编码
            codeword = encoder.encode(message)
            
            # 传输
            llr = channel.transmit(codeword, return_llr=True)
            
            # SC解码
            start_time = time.time()
            decoded_sc = sc_decoder.decode(llr.copy())
            sc_decode_time += time.time() - start_time
            
            sc_bit_errors += np.sum(decoded_sc != message)
            sc_total_bits += K
            if not np.array_equal(decoded_sc, message):
                sc_frame_errors += 1
            
            # SCL解码
            for L in list_sizes:
                start_time = time.time()
                decoded_scl = scl_decoders[L].decode(llr.copy())
                scl_decode_time[L] += time.time() - start_time
                
                scl_bit_errors[L] += np.sum(decoded_scl != message)
                scl_total_bits[L] += K
                if not np.array_equal(decoded_scl, message):
                    scl_frame_errors[L] += 1
            
            frame_count += 1
            
            # 检查是否达到最大错误数
            if sc_frame_errors >= max_errors and all(
                scl_frame_errors[L] >= max_errors for L in list_sizes
            ):
                print(f" (在第 {frame_count} 帧达到最大错误数)")
                break
        
        print(f" - 已处理 {frame_count} 帧", end='')
        
        # 计算性能指标
        sc_ber = sc_bit_errors / sc_total_bits if sc_total_bits > 0 else 1.0
        sc_fer = sc_frame_errors / frame_count if frame_count > 0 else 1.0
        sc_time_per_frame = sc_decode_time / frame_count if frame_count > 0 else 0
        
        results['sc']['ber'].append(sc_ber)
        results['sc']['fer'].append(sc_fer)
        results['sc']['time'].append(sc_time_per_frame * 1000)  # ms
        
        print(f" | SC: BER={sc_ber:.2e}, FER={sc_fer:.4f}")
        
        for L in list_sizes:
            scl_ber = scl_bit_errors[L] / scl_total_bits[L] if scl_total_bits[L] > 0 else 1.0
            scl_fer = scl_frame_errors[L] / frame_count if frame_count > 0 else 1.0
            scl_time_per_frame = scl_decode_time[L] / frame_count if frame_count > 0 else 0
            
            results['scl'][L]['ber'].append(scl_ber)
            results['scl'][L]['fer'].append(scl_fer)
            results['scl'][L]['time'].append(scl_time_per_frame * 1000)  # ms
            
            print(f"         SCL(L={L:2d}): BER={scl_ber:.2e}, FER={scl_fer:.4f}")
    
    return results


def plot_sc_scl_comparison(results: Dict, output_dir: Path):
    """绘制SC vs SCL对比图表 (5张完整报告)"""
    
    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"保存图表到目录: {output_dir.absolute()}")
    
    snr_db = np.array(results['snr_db'])
    list_sizes = list(results['scl'].keys())
    
    # 颜色方案
    colors = plt.cm.tab20(np.linspace(0, 1, len(list_sizes) + 1))
    
    # 图1: BER对比
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.semilogy(snr_db, results['sc']['ber'], 'o-', linewidth=3, markersize=8,
               label='SC (L=1)', color=colors[0], markeredgecolor='black', markeredgewidth=1.5)
    
    for i, L in enumerate(list_sizes):
        ax.semilogy(snr_db, results['scl'][L]['ber'], 's-', linewidth=2.5, markersize=7,
                   label=f'SCL (L={L})', color=colors[i+1], alpha=0.85,
                   markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel('SNR (dB)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=13, fontweight='bold')
    ax.set_title(f'BER Comparison: SC vs SCL (N={results["N"]}, K={results["K"]}, '
                f'Rate={results["rate"]:.3f})', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best', ncol=2, framealpha=0.95)
    ax.set_ylim([1e-6, 1])
    
    plt.tight_layout()
    try:
        save_path = output_dir / 'ber_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {save_path.absolute()}")
    except Exception as e:
        print(f"✗ 保存BER图失败: {e}")
    finally:
        plt.close()
    
    # 图2: FER对比
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.semilogy(snr_db, results['sc']['fer'], 'o-', linewidth=3, markersize=8,
               label='SC (L=1)', color=colors[0], markeredgecolor='black', markeredgewidth=1.5)
    
    for i, L in enumerate(list_sizes):
        ax.semilogy(snr_db, results['scl'][L]['fer'], 's-', linewidth=2.5, markersize=7,
                   label=f'SCL (L={L})', color=colors[i+1], alpha=0.85,
                   markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel('SNR (dB)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frame Error Rate (FER)', fontsize=13, fontweight='bold')
    ax.set_title(f'FER Comparison: SC vs SCL (N={results["N"]}, K={results["K"]}, '
                f'Rate={results["rate"]:.3f})', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best', ncol=2, framealpha=0.95)
    ax.set_ylim([1e-6, 1])
    
    plt.tight_layout()
    try:
        save_path = output_dir / 'fer_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {save_path.absolute()}")
    except Exception as e:
        print(f"✗ 保存FER图失败: {e}")
    finally:
        plt.close()
    
    # 图3: 解码时间对比
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(snr_db, results['sc']['time'], 'o-', linewidth=3, markersize=8,
           label='SC (L=1)', color=colors[0], markeredgecolor='black', markeredgewidth=1.5)
    
    for i, L in enumerate(list_sizes):
        ax.plot(snr_db, results['scl'][L]['time'], 's-', linewidth=2.5, markersize=7,
               label=f'SCL (L={L})', color=colors[i+1], alpha=0.85,
               markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel('SNR (dB)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Decoding Time (ms/frame)', fontsize=13, fontweight='bold')
    ax.set_title(f'Decoding Time Comparison (N={results["N"]}, K={results["K"]}, '
                f'Rate={results["rate"]:.3f})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best', ncol=2, framealpha=0.95)
    
    plt.tight_layout()
    try:
        save_path = output_dir / 'time_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {save_path.absolute()}")
    except Exception as e:
        print(f"✗ 保存时间对比图失败: {e}")
    finally:
        plt.close()
    
    # 图4: 性能改进 (相对SC的FER降低)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, L in enumerate(list_sizes):
        improvement = []
        for sc_fer, scl_fer in zip(results['sc']['fer'], results['scl'][L]['fer']):
            if sc_fer > 0:
                impr = (sc_fer - scl_fer) / sc_fer * 100
                improvement.append(max(0, impr))
            else:
                improvement.append(0)
        
        ax.plot(snr_db, improvement, 's-', linewidth=2.5, markersize=7,
               label=f'L={L}', color=colors[i+1], alpha=0.85,
               markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel('SNR (dB)', fontsize=13, fontweight='bold')
    ax.set_ylabel('FER Improvement vs SC (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Performance Gain of SCL over SC (N={results["N"]}, K={results["K"]})',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best', ncol=2, framealpha=0.95)
    ax.set_ylim([-5, 105])
    
    plt.tight_layout()
    try:
        save_path = output_dir / 'improvement_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {save_path.absolute()}")
    except Exception as e:
        print(f"✗ 保存改进对比图失败: {e}")
    finally:
        plt.close()
    
    # 图5: 复杂度vs性能 (速度-精度权衡)
    fig, ax = plt.subplots(figsize=(12, 7))

    snr_idx = 3  # 数据据有一定参考价值的SNR点
    print(f"选择的 SNR 索引: {snr_idx}, SNR={snr_db[snr_idx]} dB")

    sizes = [1] + list_sizes
    times = [float(results['sc']['time'][snr_idx])] + \
            [float(results['scl'][L]['time'][snr_idx]) for L in list_sizes]
    fers  = [max(float(results['sc']['fer'][snr_idx]), 1e-6)] + \
            [max(float(results['scl'][L]['fer'][snr_idx]), 1e-6) for L in list_sizes]

    print("sizes:", sizes)
    print("times:", times)
    print("fers :", fers)

    scatter = ax.scatter(times, fers, s=[400]*len(sizes), alpha=0.7,
                        c=range(len(sizes)), cmap='tab20', 
                        edgecolor='black', linewidth=2)

    # 添加标签
    for i, (t, f, s) in enumerate(zip(times, fers, sizes)):
        label = 'SC' if i == 0 else f'L={s}'
        print(f"点 {i}: time={t}, FER={f}, label={label}")
        ax.annotate(label, (t, f), fontsize=12, fontweight='bold',
                    xytext=(8, 8), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    ax.set_xlabel('Decoding Time (ms/frame)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frame Error Rate (FER)', fontsize=13, fontweight='bold')
    ax.set_title(f'Speed-Accuracy Trade-off (SNR={snr_db[snr_idx]:.1f} dB)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    ax.set_ylim([1e-6, 1])  # 确保 log scale 下显示

    plt.tight_layout()
    try:
        save_path = output_dir / 'complexity_tradeoff.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {save_path.absolute()}")
    except Exception as e:
        print(f"✗ 保存复杂度权衡图失败: {e}")
    finally:
        plt.close()



def generate_summary_table(results: Dict, output_dir: Path):
    """生成总结表格"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    snr_db = np.array(results['snr_db'])
    list_sizes = list(results['scl'].keys())
    
    # 创建总结
    summary = {
        'parameters': {
            'N': results['N'],
            'K': results['K'],
            'rate': results['rate'],
            'snr_range': f"{snr_db[0]:.1f} to {snr_db[-1]:.1f} dB"
        },
        'performance_metrics': {}
    }
    
    # 在各个SNR点的性能
    for snr_val in snr_db:
        idx = list(snr_db).index(snr_val)
        summary['performance_metrics'][f'SNR_{snr_val:.1f}dB'] = {
            'SC': {
                'BER': f"{results['sc']['ber'][idx]:.6e}",
                'FER': f"{results['sc']['fer'][idx]:.6f}",
                'Time_ms': f"{results['sc']['time'][idx]:.3f}"
            },
            'SCL': {}
        }
        
        for L in list_sizes:
            summary['performance_metrics'][f'SNR_{snr_val:.1f}dB']['SCL'][f'L={L}'] = {
                'BER': f"{results['scl'][L]['ber'][idx]:.6e}",
                'FER': f"{results['scl'][L]['fer'][idx]:.6f}",
                'Time_ms': f"{results['scl'][L]['time'][idx]:.3f}"
            }
    
    # 保存为JSON
    with open(output_dir / 'sc_scl_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 已保存: sc_scl_summary.json")


def full_comparison(N: int = 128, K: int = 64, 
                    snr_start: float = -1.0, snr_stop: float = 4.0, snr_step: float = 0.5,
                    list_sizes: List[int] = None,
                    num_frames: int = 200, max_errors: int = 100):
    """
    完整对比模式 - 多个SNR点的完整性能对比
    
    Args:
        N: 码长
        K: 信息位长度
        snr_start: SNR范围起点
        snr_stop: SNR范围终点
        snr_step: SNR步长
        list_sizes: SCL列表大小列表
        num_frames: 每个SNR点的最大帧数
        max_errors: 达到此错误数后提前停止
    """
    
    if list_sizes is None:
        list_sizes = [1, 2, 4, 8, 16]
    
    snr_range = np.arange(snr_start, snr_stop + snr_step/2, snr_step)
    
    # 使用绝对路径
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / 'results' / 'sc_scl_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n输出目录: {output_dir.absolute()}")
    
    # 运行仿真
    results = simulate_sc_vs_scl(
        N=N,
        K=K,
        snr_range=snr_range,
        list_sizes=list_sizes,
        num_frames=num_frames,
        max_errors=max_errors
    )
    
    # 保存原始数据
    print(f"\n生成报告...")
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ 已保存: results.json")
    
    # 生成可视化
    print(f"\n生成可视化图表...")
    plot_sc_scl_comparison(results, output_dir)
    
    # 生成总结
    generate_summary_table(results, output_dir)
    
    print(f"\n{'='*70}")
    print("✓ 完整对比完成!")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}\n")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数 - 支持命令行参数"""
    
    parser = argparse.ArgumentParser(
        description='SC vs SCL Polar Decoder Comparison Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速演示 (单个SNR点)
  python quick_sc_vs_scl.py --mode quick
  
  # 完整对比 (多个SNR点)
  python quick_sc_vs_scl.py --mode full
  
  # 自定义参数
  python quick_sc_vs_scl.py --mode full --N 256 --K 128 --snr-start -2 --snr-stop 6
        """
    )
    
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['quick', 'full'],
                       help='运行模式: quick (快速演示) 或 full (完整对比)')
    
    parser.add_argument('--N', type=int, default=128,
                       help='码长 (必须是2的幂次)')
    parser.add_argument('--K', type=int, default=64,
                       help='信息位长度')
    
    # 快速演示模式参数
    parser.add_argument('--snr', type=float, default=1.5,
                       help='快速演示模式下的SNR (dB)')
    parser.add_argument('--trials', type=int, default=500,
                       help='快速演示模式下的试验次数')
    
    # 完整对比模式参数
    parser.add_argument('--snr-start', type=float, default=-1.0,
                       help='完整对比模式下的SNR范围起点')
    parser.add_argument('--snr-stop', type=float, default=4.0,
                       help='完整对比模式下的SNR范围终点')
    parser.add_argument('--snr-step', type=float, default=0.5,
                       help='完整对比模式下的SNR步长')
    parser.add_argument('--list-sizes', type=str, default='1,2,4,8,16',
                       help='SCL列表大小 (逗号分隔)')
    parser.add_argument('--frames', type=int, default=200,
                       help='完整对比模式下每SNR点的最大帧数')
    parser.add_argument('--max-errors', type=int, default=100,
                       help='达到此错误数后提前停止')
    
    args = parser.parse_args()
    
    # 打印启动信息
    print("\n" + "="*70)
    print("SC vs SCL Polar Decoder Comparison Tool")
    print("="*70)
    
    if args.mode == 'quick':
        # 快速演示模式
        print(f"\n模式: 快速演示 (Quick Demo)")
        quick_demo(N=args.N, K=args.K, snr_db=args.snr, n_trials=args.trials)
    
    else:  # full
        # 完整对比模式
        print(f"\n模式: 完整对比 (Full Comparison)")
        list_sizes = [int(x.strip()) for x in args.list_sizes.split(',')]
        full_comparison(
            N=args.N,
            K=args.K,
            snr_start=args.snr_start,
            snr_stop=args.snr_stop,
            snr_step=args.snr_step,
            list_sizes=list_sizes,
            num_frames=args.frames,
            max_errors=args.max_errors
        )


if __name__ == "__main__":
    main()