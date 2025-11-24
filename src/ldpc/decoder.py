"""
LDPC Decoder Implementation

实现BP (Belief Propagation) 和 MS (Min-Sum) 解码算法
"""

import numpy as np
from typing import Optional, Tuple, List


class BPDecoder:
    """
    Belief Propagation (置信传播) 解码器
    
    使用和积算法进行迭代解码
    """
    
    def __init__(self, H: np.ndarray, max_iter: int = 50, early_stop: bool = True):
        """
        初始化BP解码器
        
        Args:
            H: 校验矩阵 (m x n)
            max_iter: 最大迭代次数
            early_stop: 是否在满足校验条件时提前停止
        """
        self.H = H
        self.m, self.n = H.shape
        self.max_iter = max_iter
        self.early_stop = early_stop
        
        # 构建Tanner图的邻接表
        self._build_tanner_graph()
    
    def _build_tanner_graph(self):
        """构建Tanner图的邻接表结构"""
        # 变量节点的邻居（连接的校验节点）
        self.var_neighbors = [[] for _ in range(self.n)]
        # 校验节点的邻居（连接的变量节点）
        self.check_neighbors = [[] for _ in range(self.m)]
        
        # 扫描H矩阵构建邻接表
        for i in range(self.m):
            for j in range(self.n):
                if self.H[i, j] == 1:
                    self.check_neighbors[i].append(j)
                    self.var_neighbors[j].append(i)
        
        # 预构建索引映射表，避免每次迭代中使用.index()进行线性搜索
        # var_to_check_idx[v][c] = v的邻居列表中c的索引
        self.var_to_check_idx = [{} for _ in range(self.n)]
        for v in range(self.n):
            for idx, c in enumerate(self.var_neighbors[v]):
                self.var_to_check_idx[v][c] = idx
        
        # check_to_var_idx[c][v] = c的邻居列表中v的索引
        self.check_to_var_idx = [{} for _ in range(self.m)]
        for c in range(self.m):
            for idx, v in enumerate(self.check_neighbors[c]):
                self.check_to_var_idx[c][v] = idx
    
    def _check_node_update(self, messages_in: List[float]) -> List[float]:
        """
        校验节点更新（和积算法）
        
        对于校验节点c，发送给变量节点v的消息：
        m_{c->v} = 2 * atanh(prod(tanh(m_{v'->c}/2)))  for all v' != v
        
        Args:
            messages_in: 从变量节点收到的消息列表
            
        Returns:
            发送给各变量节点的消息列表
        """
        n_neighbors = len(messages_in)
        messages_out = np.zeros(n_neighbors)
        
        # 计算所有tanh值
        tanh_values = np.tanh(np.array(messages_in) / 2.0)
        
        # 避免数值问题
        tanh_values = np.clip(tanh_values, -0.999999, 0.999999)
        
        # 对于每个输出消息，计算除了对应输入外所有tanh的乘积
        for i in range(n_neighbors):
            # 除了第i个外的所有元素的乘积
            prod = np.prod(tanh_values[np.arange(n_neighbors) != i])
            prod = np.clip(prod, -0.999999, 0.999999)
            
            # 反tanh
            messages_out[i] = 2.0 * np.arctanh(prod)
        
        # 处理无穷大和NaN
        messages_out = np.nan_to_num(messages_out, nan=0.0, posinf=20.0, neginf=-20.0)
        
        return messages_out
    
    def _variable_node_update(self, llr_channel: float, messages_in: List[float]) -> Tuple[List[float], float]:
        """
        变量节点更新
        
        对于变量节点v，发送给校验节点c的消息：
        m_{v->c} = LLR_channel + sum(m_{c'->v})  for all c' != c
        
        Args:
            llr_channel: 信道LLR
            messages_in: 从校验节点收到的消息列表
            
        Returns:
            (发送给各校验节点的消息列表, 总LLR)
        """
        n_neighbors = len(messages_in)
        messages_out = np.zeros(n_neighbors)
        
        # 总消息 = 信道LLR + 所有输入消息
        total_llr = llr_channel + np.sum(messages_in)
        
        # 对于每个输出，减去对应的输入
        for i in range(n_neighbors):
            messages_out[i] = total_llr - messages_in[i]
        
        return messages_out, total_llr
    
    def decode(self, llr: np.ndarray, return_iterations: bool = False):
        """
        BP解码
        
        Args:
            llr: 接收到的LLR值，长度为n
            return_iterations: 是否返回实际迭代次数
            
        Returns:
            解码后的比特（长度为n），如果return_iterations=True则返回(bits, iterations)
        """
        assert len(llr) == self.n, f"LLR length must be {self.n}"
        
        # 初始化消息
        # v2c[i][j]: 变量节点i发送给其第j个邻居校验节点的消息
        v2c = [np.zeros(len(self.var_neighbors[i])) for i in range(self.n)]
        # c2v[i][j]: 校验节点i发送给其第j个邻居变量节点的消息
        c2v = [np.zeros(len(self.check_neighbors[i])) for i in range(self.m)]
        
        # 初始化变量节点到校验节点的消息为信道LLR
        for v in range(self.n):
            for j in range(len(self.var_neighbors[v])):
                v2c[v][j] = llr[v]
        
        # 迭代
        actual_iterations = self.max_iter
        for iteration in range(self.max_iter):
            # 1. 校验节点更新
            for c in range(self.m):
                neighbors = self.check_neighbors[c]
                n_neighbors = len(neighbors)
                
                # 收集从变量节点的消息（使用预构建的索引映射）
                messages_in = []
                for idx, v in enumerate(neighbors):
                    c_idx_in_v = self.var_to_check_idx[v][c]
                    messages_in.append(v2c[v][c_idx_in_v])
                
                # 更新消息
                messages_out = self._check_node_update(messages_in)
                
                # 存储消息（使用预构建的索引映射）
                for idx, v in enumerate(neighbors):
                    c_idx_in_v = self.var_to_check_idx[v][c]
                    c2v[c][idx] = messages_out[idx]
            
            # 2. 变量节点更新
            total_llrs = np.zeros(self.n)
            
            for v in range(self.n):
                neighbors = self.var_neighbors[v]
                
                # 收集从校验节点的消息（使用预构建的索引映射）
                messages_in = []
                for c in neighbors:
                    v_idx_in_c = self.check_to_var_idx[c][v]
                    messages_in.append(c2v[c][v_idx_in_c])
                
                # 更新消息
                messages_out, total_llr = self._variable_node_update(llr[v], messages_in)
                total_llrs[v] = total_llr
                
                # 存储消息
                for idx, c in enumerate(neighbors):
                    v2c[v][idx] = messages_out[idx]
            
            # 3. 硬判决
            decoded = (total_llrs <= 0).astype(int)
            
            # 4. 提前停止检查
            if self.early_stop:
                syndrome = (self.H @ decoded) % 2
                if np.all(syndrome == 0):
                    actual_iterations = iteration + 1
                    break
        
        if return_iterations:
            return decoded, actual_iterations
        return decoded
    
    def __repr__(self) -> str:
        return f"BPDecoder(n={self.n}, m={self.m}, max_iter={self.max_iter})"


class MSDecoder:
    """
    Min-Sum 解码器
    
    BP算法的简化版本，使用最小值代替tanh/atanh运算
    """
    
    def __init__(self, H: np.ndarray, max_iter: int = 50, 
                 normalization: float = 1.0, early_stop: bool = True):
        """
        初始化MS解码器
        
        Args:
            H: 校验矩阵
            max_iter: 最大迭代次数
            normalization: 归一化因子（用于Normalized MS）
            early_stop: 是否提前停止
        """
        self.H = H
        self.m, self.n = H.shape
        self.max_iter = max_iter
        self.normalization = normalization
        self.early_stop = early_stop
        
        # 构建Tanner图
        self._build_tanner_graph()
    
    def _build_tanner_graph(self):
        """构建Tanner图"""
        self.var_neighbors = [[] for _ in range(self.n)]
        self.check_neighbors = [[] for _ in range(self.m)]
        
        for i in range(self.m):
            for j in range(self.n):
                if self.H[i, j] == 1:
                    self.check_neighbors[i].append(j)
                    self.var_neighbors[j].append(i)
        
        # 预构建索引映射表，避免每次迭代中使用.index()进行线性搜索
        self.var_to_check_idx = [{} for _ in range(self.n)]
        for v in range(self.n):
            for idx, c in enumerate(self.var_neighbors[v]):
                self.var_to_check_idx[v][c] = idx
        
        self.check_to_var_idx = [{} for _ in range(self.m)]
        for c in range(self.m):
            for idx, v in enumerate(self.check_neighbors[c]):
                self.check_to_var_idx[c][v] = idx
    
    def _check_node_update_ms(self, messages_in: List[float]) -> List[float]:
        """
        校验节点更新（Min-Sum）
        
        m_{c->v} = sign(prod(sign(m_{v'->c}))) * min(|m_{v'->c}|)
        
        Args:
            messages_in: 输入消息
            
        Returns:
            输出消息
        """
        n_neighbors = len(messages_in)
        messages_out = np.zeros(n_neighbors)
        messages_in = np.array(messages_in)
        
        # 计算符号和绝对值
        signs = np.sign(messages_in)
        abs_vals = np.abs(messages_in)
        
        for i in range(n_neighbors):
            # 符号：除了第i个外所有符号的乘积
            sign_prod = np.prod(signs[np.arange(n_neighbors) != i])
            
            # 幅度：除了第i个外的最小值
            min_val = np.min(abs_vals[np.arange(n_neighbors) != i])
            
            # 归一化
            messages_out[i] = sign_prod * min_val * self.normalization
        
        return messages_out
    
    def decode(self, llr: np.ndarray) -> np.ndarray:
        """
        Min-Sum解码
        
        Args:
            llr: 接收LLR
            
        Returns:
            解码比特
        """
        assert len(llr) == self.n, f"LLR length must be {self.n}"
        
        # 初始化消息
        v2c = [np.zeros(len(self.var_neighbors[i])) for i in range(self.n)]
        c2v = [np.zeros(len(self.check_neighbors[i])) for i in range(self.m)]
        
        # 初始化
        for v in range(self.n):
            for j in range(len(self.var_neighbors[v])):
                v2c[v][j] = llr[v]
        
        # 迭代
        for iteration in range(self.max_iter):
            # 校验节点更新
            for c in range(self.m):
                neighbors = self.check_neighbors[c]
                
                messages_in = []
                for v in neighbors:
                    c_idx_in_v = self.var_to_check_idx[v][c]
                    messages_in.append(v2c[v][c_idx_in_v])
                
                messages_out = self._check_node_update_ms(messages_in)
                
                for idx, v in enumerate(neighbors):
                    c2v[c][idx] = messages_out[idx]
            
            # 变量节点更新
            total_llrs = np.zeros(self.n)
            
            for v in range(self.n):
                neighbors = self.var_neighbors[v]
                
                messages_in = []
                for c in neighbors:
                    v_idx_in_c = self.check_to_var_idx[c][v]
                    messages_in.append(c2v[c][v_idx_in_c])
                
                total_llr = llr[v] + np.sum(messages_in)
                total_llrs[v] = total_llr
                
                for idx, c in enumerate(neighbors):
                    v2c[v][idx] = total_llr - messages_in[idx]
            
            # 硬判决
            decoded = (total_llrs <= 0).astype(int)
            
            # 提前停止
            if self.early_stop:
                syndrome = (self.H @ decoded) % 2
                if np.all(syndrome == 0):
                    break
        
        return decoded
    
    def __repr__(self) -> str:
        return f"MSDecoder(n={self.n}, m={self.m}, max_iter={self.max_iter}, norm={self.normalization})"


if __name__ == "__main__":
    # 测试代码
    print("Testing LDPC Decoders...")
    
    from .encoder import LDPCEncoder
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from channel import AWGNChannel
    
    # 测试1: BP解码器（无噪声）
    print("\n1. BP Decoder Test (No Noise):")
    n, k = 24, 12
    encoder = LDPCEncoder(n, k, dv=3, dc=6, seed=42)
    decoder = BPDecoder(encoder.H, max_iter=50)
    
    print(f"Encoder: {encoder}")
    print(f"Decoder: {decoder}")
    
    message = np.random.randint(0, 2, k)
    codeword = encoder.encode(message)
    
    # 无噪声，高SNR LLR
    llr = (1 - 2 * codeword.astype(float)) * 100
    decoded = decoder.decode(llr)
    
    # 提取信息位
    decoded_msg = decoded[:k]
    
    print(f"Message: {message}")
    print(f"Decoded: {decoded_msg}")
    print(f"Correct: {np.array_equal(message, decoded_msg)}")
    
    # 测试2: BP解码器（有噪声）
    print("\n2. BP Decoder Test (With AWGN):")
    n, k = 48, 24
    snr_db = 3.0
    
    encoder = LDPCEncoder(n, k, dv=3, dc=6, seed=42)
    decoder = BPDecoder(encoder.H, max_iter=50)
    channel = AWGNChannel(snr_db=snr_db, seed=42)
    
    num_tests = 10
    errors = 0
    
    for i in range(num_tests):
        msg = np.random.randint(0, 2, k)
        cw = encoder.encode(msg)
        llr = channel.transmit(cw, return_llr=True)
        decoded = decoder.decode(llr)
        decoded_msg = decoded[:k]
        
        if not np.array_equal(msg, decoded_msg):
            errors += 1
    
    print(f"SNR = {snr_db} dB")
    print(f"Frame errors: {errors}/{num_tests}")
    print(f"FER: {errors/num_tests:.4f}")
    
    # 测试3: MS解码器
    print("\n3. MS Decoder Test:")
    n, k = 24, 12
    encoder = LDPCEncoder(n, k, dv=3, dc=6, seed=42)
    ms_decoder = MSDecoder(encoder.H, max_iter=50, normalization=0.75)
    
    print(f"Decoder: {ms_decoder}")
    
    msg = np.random.randint(0, 2, k)
    cw = encoder.encode(msg)
    llr = (1 - 2 * cw.astype(float)) * 100
    decoded = ms_decoder.decode(llr)
    decoded_msg = decoded[:k]
    
    print(f"Message: {msg}")
    print(f"Decoded: {decoded_msg}")
    print(f"Correct: {np.array_equal(msg, decoded_msg)}")
    
    # 测试4: BP vs MS对比
    print("\n4. BP vs MS Comparison:")
    n, k = 48, 24
    snr_db = 2.0
    
    encoder = LDPCEncoder(n, k, dv=3, dc=6, seed=42)
    bp_decoder = BPDecoder(encoder.H, max_iter=50)
    ms_decoder = MSDecoder(encoder.H, max_iter=50, normalization=0.75)
    channel = AWGNChannel(snr_db=snr_db, seed=42)
    
    num_tests = 20
    bp_errors = 0
    ms_errors = 0
    
    for i in range(num_tests):
        msg = np.random.randint(0, 2, k)
        cw = encoder.encode(msg)
        llr = channel.transmit(cw, return_llr=True)
        
        # BP解码
        decoded_bp = bp_decoder.decode(llr)[:k]
        if not np.array_equal(msg, decoded_bp):
            bp_errors += 1
        
        # MS解码
        decoded_ms = ms_decoder.decode(llr)[:k]
        if not np.array_equal(msg, decoded_ms):
            ms_errors += 1
    
    print(f"SNR = {snr_db} dB, Frames = {num_tests}")
    print(f"BP FER: {bp_errors/num_tests:.4f}")
    print(f"MS FER: {ms_errors/num_tests:.4f}")
    
    print("\n✓ LDPC Decoder test passed!")
