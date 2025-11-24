"""
Backup: Hard-decision based Polar decoders (SC + SCL wrapper).

这是当前已经验证在无噪声下可逆、在AWGN下表现正常的解码实现备份。
后续如需对SC/SCL进行实验性改动，可以随时从这里恢复。
"""

import numpy as np
from typing import Optional

from .utils import polar_transform_iterative


class SCDecoder:
    """Hard-decision + inverse polar transform decoder.

    思路：
    - 对接收LLR先做硬判决得到码字比特 x_hat
    - 由于极化矩阵 G_N = F^{\otimes n} 在GF(2)上满足 G_N^{-1} = G_N，
      对 x_hat 再做一次 polar_transform_iterative 即可恢复 u_hat
    - 然后取信息位位置 self.info_bits 作为解码结果

    在无噪声场景下，该解码器与 PolarEncoder 完全互逆；
    在一般AWGN信道下，相当于先做BSC硬判决，再做线性逆变换，
    性能稳定且实现简单，适合作为基线和备份。
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

        # 冻结位掩码
        self.frozen_mask = np.zeros(N, dtype=bool)
        self.frozen_mask[self.frozen_bits] = True

    def decode(self, llr_input: np.ndarray) -> np.ndarray:
        """Decode using hard decision + inverse polar transform."""
        llr_input = np.asarray(llr_input, dtype=float)
        assert llr_input.shape == (self.N,), f"expected LLR shape ({self.N},), got {llr_input.shape}"

        # 1) 硬判决到码字比特 (0/1)
        x_hat = (llr_input < 0).astype(int)

        # 2) 利用 G_N^2 = I，在GF(2)上再次做Polar变换即可得到 u_hat
        u_hat = polar_transform_iterative(x_hat)

        # 3) 冻结位强制为0（理论上若编码正确，这里本来就应为0）
        u_hat[self.frozen_bits] = 0

        # 4) 返回信息位
        return u_hat[self.info_bits]

    # 递归版本保留作实验用途
    def _decode_recursive(self, llr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        N = llr.size
        assert mask.size == N
        if N == 1:
            if mask[0]:
                return np.array([0], dtype=int)
            return np.array([0 if llr[0] >= 0 else 1], dtype=int)
        half = N // 2
        llr1 = llr[:half]
        llr2 = llr[half:]
        mask1 = mask[:half]
        mask2 = mask[half:]
        sign_prod = np.sign(llr1) * np.sign(llr2)
        min_abs = np.minimum(np.abs(llr1), np.abs(llr2))
        llr_left = sign_prod * min_abs
        u_left = self._decode_recursive(llr_left, mask1)
        llr_right = llr2 + (1.0 - 2.0 * u_left.astype(float)) * llr1
        u_right = self._decode_recursive(llr_right, mask2)
        return np.concatenate([u_left, u_right])

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"SCDecoder_HD(N={self.N}, K={self.K})"


class SCLDecoder:
    """Simple SCL wrapper around the hard-decision SC decoder.

    当前实现：无论 list_size，行为均等价于 SCDecoder（L=1 的 SCL）。
    作为备份实现保留，便于对比或在实验性改动出问题时快速回退。
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

        self._sc = SCDecoder(N, K, frozen_bits=frozen_bits)

    def decode(self, llr_input: np.ndarray) -> np.ndarray:
        return self._sc.decode(llr_input)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"SCLDecoder_HD(N={self.N}, K={self.K}, L={self.L}, use_crc={self.use_crc})"