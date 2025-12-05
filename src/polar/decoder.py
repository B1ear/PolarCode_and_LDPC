"""
Polar码解码器

SC (连续消除) 解码器使用标准的迭代比特反转算法。
SCL (连续消除列表) 解码器维护多个候选路径。
"""

import numpy as np
from typing import Optional

from .utils import bit_reverse
class SCDecoder:
    """标准迭代SC解码器
    """

    def __init__(self, N: int, K: int, frozen_bits: Optional[np.ndarray] = None):
        assert N > 0 and (N & (N - 1)) == 0, "N must be a power of 2"
        assert 0 < K < N, "K must be in (0, N)"

        self.N = N
        self.K = K
        self.n = int(np.log2(N))  # 树深度

        if frozen_bits is None:
            from .utils import generate_frozen_bits
            self.frozen_bits, self.info_bits = generate_frozen_bits(N, K)
        else:
            self.frozen_bits = np.array(frozen_bits, dtype=int)
            self.info_bits = np.setdiff1d(np.arange(N), self.frozen_bits)
        
        # 将冻结位转换为集合以实现O(1)查找
        self.frozen_set = set(self.frozen_bits)
        
        # LLR和比特矩阵 - 用NaN初始化
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan, dtype=np.float64)

    def decode(self, llr_input: np.ndarray) -> np.ndarray:
        """SC解码
        
        Args:
            llr_input: 信道LLR值，长度为N（正值表示比特=0更可能）
        
        Returns:
            解码后的信息位，长度为K
        """
        llr_input = np.asarray(llr_input, dtype=np.float64)
        assert llr_input.shape == (self.N,), f"expected LLR shape ({self.N},), got {llr_input.shape}"

        # 用信道观测值初始化LLR矩阵
        self.L[:, 0] = llr_input
        
        # 按比特反转顺序解码
        for i in range(self.N):
            l = bit_reverse(i, self.n)
            
            # 更新LLR
            self._update_llrs(l)
            
            # 做硬判决
            if l in self.frozen_set:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = self._hard_decision(self.L[l, self.n])
            
            # 传播比特判决
            self._update_bits(l)
        
        # 提取信息位
        u_decoded = self.B[:, self.n].astype(int)
        return u_decoded[self.info_bits]

    def _update_llrs(self, l: int):
        """更新LLR 
        
        Args:
            l: 要更新LLR的比特索引
        """
        # 遍历相关深度
        for s in range(self.n - self._active_llr_level(l, self.n), self.n):
            block_size = int(2 ** (s + 1))
            branch_size = int(block_size / 2)
            
            # 遍历此深度的所有位置
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:  # 上分支
                    top_llr = self.L[j, s]
                    btm_llr = self.L[j + branch_size, s]
                    self.L[j, s + 1] = self._upper_llr(top_llr, btm_llr)
                else:  # 下分支
                    btm_llr = self.L[j, s]
                    top_llr = self.L[j - branch_size, s]
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = self._lower_llr(btm_llr, top_llr, top_bit)
    
    def _update_bits(self, l: int):
        """更新比特
        
        Args:
            l: 要更新比特的索引
        """
        # 上半部分提前返回
        if l < self.N / 2:
            return
        
        # 从n向下遍历到活跃层级
        for s in range(self.n, self.n - self._active_bit_level(l, self.n), -1):
            block_size = int(2 ** s)
            branch_size = int(block_size / 2)
            
            # 从l向后遍历
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:  # 下分支
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    self.B[j, s - 1] = self.B[j, s]
    
    def _hard_decision(self, y: float) -> int:
        """对LLR进行硬判决。"""
        return 0 if y >= 0 else 1
    
    def _upper_llr(self, l1: float, l2: float) -> float:
        """上分支LLR更新（f函数）。
        
        简化版本，不使用对数域运算。
        使用最小和近似：f(a,b) = sign(a)*sign(b)*min(|a|,|b|)
        """
        return np.sign(l1) * np.sign(l2) * min(abs(l1), abs(l2))
    
    def _lower_llr(self, l1: float, l2: float, b: float) -> float:
        """下分支LLR更新（g函数）。
        
        
        Args:
            l1: 底部LLR（polarcodes中的btm_llr）
            l2: 顶部LLR（polarcodes中的top_llr）
            b: 顶部比特判决
        
        Returns:
            更新后的LLR
        """
        if b == 0:
            return l1 + l2  # btm + top
        else:  # b == 1
            return l1 - l2  # btm - top
    
    def _active_llr_level(self, i: int, n: int) -> int:
        """在i的二进制展开中找到第一个1。
        """
        mask = 2 ** (n - 1)
        count = 1
        for k in range(n):
            if (mask & i) == 0:
                count += 1
                mask >>= 1
            else:
                break
        return min(count, n)
    
    def _active_bit_level(self, i: int, n: int) -> int:
        """在i的二进制展开中找到第一个0。
        """
        mask = 2 ** (n - 1)
        count = 1
        for k in range(n):
            if (mask & i) > 0:
                count += 1
                mask >>= 1
            else:
                break
        return min(count, n)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"SCDecoder(N={self.N}, K={self.K})"


class SCLDecoder:
    """连续消除列表（SCL）解码器。
    
    SCL解码器在解码过程中维护多个候选路径，
    最后选择最可能的路径。
    
    Args:
        N: 码长（必须是2的幂次）
        K: 信息位长度
        list_size: 要维护的路径数量（L）
        frozen_bits: 可选的冻结位位置
        use_crc: 是否使用CRC进行路径选择
        crc_polynomial: 要使用的CRC多项式（如果use_crc=True）
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
        self.n = int(np.log2(N))
        self.use_crc = use_crc
        self.crc_polynomial = crc_polynomial

        if frozen_bits is None:
            from .utils import generate_frozen_bits
            self.frozen_bits, self.info_bits = generate_frozen_bits(N, K)
        else:
            self.frozen_bits = np.array(frozen_bits, dtype=int)
            self.info_bits = np.setdiff1d(np.arange(N), self.frozen_bits)
        
        # 将冻结位转换为集合以实现O(1)查找
        self.frozen_set = set(self.frozen_bits)
        
        # 路径度量（对数似然）
        self.path_metrics = np.full(self.L, -np.inf)
        
        # 每个路径维护自己的LLR和比特矩阵
        self.L_paths = np.full((self.L, N, self.n + 1), np.nan, dtype=np.float64)
        self.B_paths = np.full((self.L, N, self.n + 1), np.nan, dtype=np.float64)
        
        # 活跃路径
        self.active_paths = np.zeros(self.L, dtype=bool)

    def decode(self, llr_input: np.ndarray) -> np.ndarray:
        """使用列表解码的SCL解码。
        
        Args:
            llr_input: 信道LLR值，长度为N
        
        Returns:
            解码后的信息位，长度为K
        """
        llr_input = np.asarray(llr_input, dtype=np.float64)
        assert llr_input.shape == (self.N,), f"expected LLR shape ({self.N},), got {llr_input.shape}"

        # 初始化：一个度量为0的活跃路径
        self.active_paths[:] = False
        self.active_paths[0] = True
        self.path_metrics[:] = -np.inf
        self.path_metrics[0] = 0.0
        
        # 用信道LLR初始化所有路径
        for l in range(self.L):
            self.L_paths[l, :, 0] = llr_input
        
        # 按比特反转顺序解码
        for i in range(self.N):
            l = bit_reverse(i, self.n)
            
            if l in self.frozen_set:
                # 冻结位：强制为0，不分裂路径
                self._decode_frozen_bit(l)
            else:
                # 信息位：分裂路径
                self._decode_info_bit(l)
        
        # 选择最佳路径
        best_path_idx = np.argmax(self.path_metrics)
        u_decoded = self.B_paths[best_path_idx, :, self.n].astype(int)
        
        return u_decoded[self.info_bits]

    def _decode_frozen_bit(self, l: int):
        """解码冻结位（对所有活跃路径设置为0）。"""
        for path_idx in range(self.L):
            if not self.active_paths[path_idx]:
                continue
            
            # 更新此路径的LLR
            self._update_llrs_for_path(path_idx, l)
            
            # 将冻结位设置为0
            self.B_paths[path_idx, l, self.n] = 0
            
            # 更新路径度量
            llr_val = self.L_paths[path_idx, l, self.n]
            self.path_metrics[path_idx] += self._log_likelihood(llr_val, 0)
            
            # 传播比特判决
            self._update_bits_for_path(path_idx, l)

    def _decode_info_bit(self, l: int):
        """解码信息位（为0和1分裂路径）。"""
        # 找到活跃路径
        active_indices = np.where(self.active_paths)[0]
        n_active = len(active_indices)
        
        # 计算两种比特选择的度量
        path_metrics_0 = []
        path_metrics_1 = []
        
        for path_idx in active_indices:
            # 更新LLR
            self._update_llrs_for_path(path_idx, l)
            llr_val = self.L_paths[path_idx, l, self.n]
            
            # bit=0和bit=1的度量
            metric_0 = self.path_metrics[path_idx] + self._log_likelihood(llr_val, 0)
            metric_1 = self.path_metrics[path_idx] + self._log_likelihood(llr_val, 1)
            
            path_metrics_0.append((metric_0, path_idx, 0))
            path_metrics_1.append((metric_1, path_idx, 1))
        
        # 合并并排序所有候选路径
        all_candidates = path_metrics_0 + path_metrics_1
        all_candidates.sort(key=lambda x: x[0], reverse=True)  # 按度量排序（降序）
        
        # 选择前L条路径
        n_survive = min(len(all_candidates), self.L)
        survivors = all_candidates[:n_survive]
        
        # 存储旧路径状态
        old_L_paths = self.L_paths.copy()
        old_B_paths = self.B_paths.copy()
        old_metrics = self.path_metrics.copy()
        
        # 重置活跃路径
        self.active_paths[:] = False
        self.path_metrics[:] = -np.inf
        
        # 更新幸存者
        for new_idx, (metric, old_idx, bit) in enumerate(survivors):
            if new_idx >= self.L:
                break
            
            # 复制路径状态
            self.L_paths[new_idx] = old_L_paths[old_idx].copy()
            self.B_paths[new_idx] = old_B_paths[old_idx].copy()
            
            # 设置比特判决
            self.B_paths[new_idx, l, self.n] = bit
            
            # 更新度量
            self.path_metrics[new_idx] = metric
            self.active_paths[new_idx] = True
            
            # 传播比特判决
            self._update_bits_for_path(new_idx, l)

    def _update_llrs_for_path(self, path_idx: int, l: int):
        """更新特定路径的LLR。"""
        for s in range(self.n - self._active_llr_level(l, self.n), self.n):
            block_size = int(2 ** (s + 1))
            branch_size = int(block_size / 2)
            
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:  # 上分支
                    top_llr = self.L_paths[path_idx, j, s]
                    btm_llr = self.L_paths[path_idx, j + branch_size, s]
                    self.L_paths[path_idx, j, s + 1] = self._upper_llr(top_llr, btm_llr)
                else:  # 下分支
                    btm_llr = self.L_paths[path_idx, j, s]
                    top_llr = self.L_paths[path_idx, j - branch_size, s]
                    top_bit = self.B_paths[path_idx, j - branch_size, s + 1]
                    self.L_paths[path_idx, j, s + 1] = self._lower_llr(btm_llr, top_llr, top_bit)

    def _update_bits_for_path(self, path_idx: int, l: int):
        """更新特定路径的比特。"""
        if l < self.N / 2:
            return
        
        for s in range(self.n, self.n - self._active_bit_level(l, self.n), -1):
            block_size = int(2 ** s)
            branch_size = int(block_size / 2)
            
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:  # 下分支
                    self.B_paths[path_idx, j - branch_size, s - 1] = (
                        int(self.B_paths[path_idx, j, s]) ^ int(self.B_paths[path_idx, j - branch_size, s])
                    )
                    self.B_paths[path_idx, j, s - 1] = self.B_paths[path_idx, j, s]

    def _log_likelihood(self, llr: float, bit: int) -> float:
        """计算对数似然贡献（用于最大化）。
        
        对于 LLR = log(P(bit=0)/P(bit=1)):
        我们要计算 log P(bit|LLR)
        
        P(bit=0|LLR) = e^LLR / (1 + e^LLR) = 1 / (1 + e^-LLR)
        P(bit=1|LLR) = 1 / (1 + e^LLR)
        
        log P(bit=0|LLR) = -log(1 + e^-LLR)
        log P(bit=1|LLR) = -log(1 + e^LLR)
        
        使用数值稳定的计算：
        -log(1 + e^x) = -log(e^max(0,x) * (e^-max(0,x) + e^(x-max(0,x))))
                      = -max(0, x) - log1p(e^(-|x|))
        """
        # 数值稳定的计算
        if bit == 0:
            # log P(bit=0) = -log(1 + e^-LLR)
            if llr >= 0:
                # 对于正LLR，e^-LLR很小
                return -np.log1p(np.exp(-llr))
            else:
                # 对于负LLR，重写为：-log(1 + e^-LLR) = -log(e^-LLR(e^LLR + 1)) = LLR - log(1 + e^LLR)
                return llr - np.log1p(np.exp(llr))
        else:
            # log P(bit=1) = -log(1 + e^LLR)
            if llr >= 0:
                # 对于正LLR，重写：-log(1 + e^LLR) = -log(e^LLR(e^-LLR + 1)) = -LLR - log(1 + e^-LLR)
                return -llr - np.log1p(np.exp(-llr))
            else:
                # 对于负LLR，e^LLR很小
                return -np.log1p(np.exp(llr))

    def _upper_llr(self, l1: float, l2: float) -> float:
        """上分支LLR更新（f函数）- 最小和近似。"""
        return np.sign(l1) * np.sign(l2) * min(abs(l1), abs(l2))

    def _lower_llr(self, l1: float, l2: float, b: float) -> float:
        """下分支LLR更新（g函数）。"""
        if b == 0:
            return l1 + l2
        else:
            return l1 - l2

    def _active_llr_level(self, i: int, n: int) -> int:
        """在i的二进制展开中找到第一个1。"""
        mask = 2 ** (n - 1)
        count = 1
        for k in range(n):
            if (mask & i) == 0:
                count += 1
                mask >>= 1
            else:
                break
        return min(count, n)

    def _active_bit_level(self, i: int, n: int) -> int:
        """在i的二进制展开中找到第一个0。"""
        mask = 2 ** (n - 1)
        count = 1
        for k in range(n):
            if (mask & i) > 0:
                count += 1
                mask >>= 1
            else:
                break
        return min(count, n)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"SCLDecoder(N={self.N}, K={self.K}, L={self.L}, use_crc={self.use_crc})"
