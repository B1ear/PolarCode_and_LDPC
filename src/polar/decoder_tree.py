"""
Experimental tree-structured soft-decision SC / SCL decoders for Polar codes.

IMPORTANT (实验性质说明):
- 这里实现的是基于递归树结构的软判决 SC / SCL 解码框架，
  目的是为后续真正的列表解码 (SCL) 打下数据结构和接口基础。
- 当前 TreeSCDecoder 的软判决逻辑尚未严格证明与编码变换完全对齐，
  仍处于实验/调试阶段；请继续使用 `decoder_hd.SCDecoder` 作为稳定基线。

后续计划：
- 用 N=8,16 等小码长做无噪声穷举测试，对齐 f/g 调度和索引，
  直到 TreeSCDecoder 在无噪声场景下也能严格可逆；
- 在 TreeSCDecoder 的 LLR/部分和树之上实现真正的 TreeSCLDecoder
  (路径分裂 + 路径度量 + 剪枝 + 可选 CRC)。
"""

import numpy as np
from typing import Optional, List, Dict


def f_min_sum(a: float, b: float) -> float:
    """Check-node update (min-sum approximation)."""
    return float(np.sign(a) * np.sign(b) * min(abs(a), abs(b)))


def g_update(a: float, b: float, u: int) -> float:
    """Variable-node update g(a,b,u) = b + (1-2u)*a."""
    return float(b + (1.0 - 2.0 * u) * a)


class TreeSCDecoder:
    """Experimental tree-structured soft-decision SC decoder.

    注意：
    - 这是一个实验性实现，目前只保证结构/接口合理，
      不保证在所有参数下都严格达到理论最优性能；
    - 建议用于对比/研究，不要立即替换主线 `decoder.SCDecoder`。
    """

    def __init__(self, N: int, K: int, frozen_bits: Optional[np.ndarray] = None):
        assert N > 0 and (N & (N - 1)) == 0, "N must be a power of 2"
        assert 0 < K < N, "K must be in (0, N)"

        self.N = N
        self.K = K

        if frozen_bits is None:
            from .utils import generate_frozen_bits
            self.frozen_bits, self.info_bits = generate_frozen_bits(N, K)
        else:
            self.frozen_bits = np.array(frozen_bits, dtype=int)
            self.info_bits = np.setdiff1d(np.arange(N), self.frozen_bits)

        self.frozen_mask = np.zeros(N, dtype=bool)
        self.frozen_mask[self.frozen_bits] = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode(self, llr_input: np.ndarray) -> np.ndarray:
        """Soft-decision SC decoding on a full codeword (experimental).

        当前实现：在树递归内部**不使用冻结位信息**，仅基于LLR做SC判决，
        然后在根节点对冻结位强制为0。这样可以先验证树状结构与编码器
        的对齐情况（无噪声穷举），后续再将冻结位完全融入递归中。

        Args:
            llr_input: shape (N,) LLRs from channel.
        Returns:
            Estimated information bits, shape (K,).
        """
        llr_input = np.asarray(llr_input, dtype=float)
        assert llr_input.shape == (self.N,)

        # 递归解整个码字对应的“本地u向量”
        u_hat_full = self._decode_node(llr_input)

        # 在顶层强制冻结位为0（编码器本来也把这些位置设为0）
        u_hat_full = u_hat_full.astype(int)
        u_hat_full[self.frozen_bits] = 0

        return u_hat_full[self.info_bits]

    # ------------------------------------------------------------------
    # Recursive tree decoding (block-based, experimental)
    # ------------------------------------------------------------------

    def _decode_node(self, llr: np.ndarray) -> np.ndarray:
        """Decode a sub-block represented by this node (no frozen bits inside).

        结构说明：
        - 该递归对应于编码递归 x = PolarTransform(u) 的“反向树”，
          先解左子块，再根据左子块的估计更新右子块的LLR，最后在父节点
          处做一次 (u_left, u_right) -> (u_left ^ u_right, u_right) 的组合，
          以恢复该子块对应的全局 u 比特。

        Args:
            llr: shape (M,) current sub-block LLRs.
        Returns:
            u_hat for this sub-block, shape (M,).
        """
        M = llr.size

        # Base case: length-1 sub-code（直接按符号判决）
        if M == 1:
            return np.array([0 if llr[0] >= 0 else 1], dtype=int)

        # Split into two halves: llr 对应编码后的 [x_left, x_right]
        half = M // 2
        llr1 = llr[:half]
        llr2 = llr[half:]

        # 左子块：使用 f 函数组合得到对应于“左输入”比特的LLR
        llr_left = np.empty_like(llr1)
        for i in range(half):
            llr_left[i] = f_min_sum(llr1[i], llr2[i])

        u_left = self._decode_node(llr_left)

        # 右子块：根据左子块估计的比特，用 g 函数更新LLR
        llr_right = np.empty_like(llr2)
        for i in range(half):
            llr_right[i] = g_update(llr1[i], llr2[i], int(u_left[i]))

        u_right = self._decode_node(llr_right)

        # 父节点上把局部比特 (v, w) 组合回原始 u:
        #   u_left_global  = v ^ w
        #   u_right_global = w
        u_hat = np.empty(M, dtype=int)
        u_hat[half:] = u_right
        u_hat[:half] = (u_left ^ u_right) & 1

        return u_hat

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"TreeSCDecoder(N={self.N}, K={self.K})"


class TreeSCLDecoder:
    """Skeleton of a tree-structured SCL decoder (to be completed).

    当前仅提供接口和数据结构雏形，后续会基于 TreeSCDecoder 的节点递归逻辑
    实现真正的路径分裂、路径度量和剪枝。
    """

    def __init__(self, N: int, K: int, list_size: int = 8,
                 frozen_bits: Optional[np.ndarray] = None,
                 use_crc: bool = False, crc_polynomial: str = "CRC-8"):
        assert N > 0 and (N & (N - 1)) == 0, "N must be a power of 2"
        assert 0 < K < N, "K must be in (0, N)"
        assert list_size >= 1

        self.N = N
        self.K = K
        self.L = list_size
        self.use_crc = use_crc
        self.crc_polynomial = crc_polynomial

        if frozen_bits is None:
            from .utils import generate_frozen_bits
            self.frozen_bits, self.info_bits = generate_frozen_bits(N, K)
        else:
            self.frozen_bits = np.array(frozen_bits, dtype=int)
            self.info_bits = np.setdiff1d(np.arange(N), self.frozen_bits)

        self.frozen_mask = np.zeros(N, dtype=bool)
        self.frozen_mask[self.frozen_bits] = True

    def decode(self, llr_input: np.ndarray) -> np.ndarray:
        """Placeholder decode using TreeSCDecoder as a single-path backend.

        真正的列表解码会在这里实现；当前仅调用 TreeSCDecoder 作为占位，
        以便先验证树状软判决结构在接口上的可行性。
        """
        from .decoder_hd import SCDecoder as HD_SC

        # 暂时直接使用备份硬判决解码器作为列表解码器的后端，
        # 保证行为稳定；后续会改成基于 _decode_node 的多路径实现。
        hd_sc = HD_SC(self.N, self.K, frozen_bits=self.frozen_bits)
        return hd_sc.decode(llr_input)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"TreeSCLDecoder(N={self.N}, K={self.K}, L={self.L}, use_crc={self.use_crc})"