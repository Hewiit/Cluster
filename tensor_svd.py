import numpy as np
import torch
import cvxpy as cp
from sklearn.metrics import silhouette_score
from config import *

class TensorSVD:
    """t-SVD张量分解模块"""
    
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
    
    def t_product(self, A, B):
        """
        t-积运算
        
        Args:
            A: 张量 A (n1 × n2 × n3)
            B: 张量 B (n2 × l × n3)
            
        Returns:
            张量 C (n1 × l × n3)
        """
        n1, n2, n3 = A.shape
        _, l, _ = B.shape
        
        # 在第三维上进行FFT
        A_fft = np.fft.fft(A, axis=2)
        B_fft = np.fft.fft(B, axis=2)
        
        # 初始化结果张量
        C = np.zeros((n1, l, n3), dtype=complex)
        
        # 对每个频率分量进行矩阵乘法
        for i in range(n3):
            C[:, :, i] = A_fft[:, :, i] @ B_fft[:, :, i]
        
        # 逆FFT
        C = np.real(np.fft.ifft(C, axis=2))
        
        return C
    
    def t_svd_decomposition(self, tensor):
        """
        t-SVD分解
        
        Args:
            tensor: 三阶张量 (n1 × n2 × n3)
            
        Returns:
            U, S, V: 分解结果
        """
        n1, n2, n3 = tensor.shape
        
        # 在第三维上进行FFT
        tensor_fft = np.fft.fft(tensor, axis=2)
        
        # 初始化结果张量
        U = np.zeros((n1, n1, n3), dtype=complex)
        S = np.zeros((n1, n2, n3), dtype=complex)
        V = np.zeros((n2, n2, n3), dtype=complex)
        
        # 对每个频率分量进行SVD
        for i in range(n3):
            u, s, vh = np.linalg.svd(tensor_fft[:, :, i], full_matrices=True)
            U[:, :, i] = u
            
            # 正确处理奇异值矩阵
            # 创建对角矩阵，确保维度正确
            s_diag = np.zeros((n1, n2))
            min_dim = min(n1, n2)
            s_diag[:min_dim, :min_dim] = np.diag(s[:min_dim])
            S[:, :, i] = s_diag
            
            V[:, :, i] = vh.T
        
        # 逆FFT
        U = np.real(np.fft.ifft(U, axis=2))
        S = np.real(np.fft.ifft(S, axis=2))
        V = np.real(np.fft.ifft(V, axis=2))
        
        return U, S, V
    
    def low_rank_approximation(self, U, S, V, rank):
        """
        基于t-SVD的低秩近似
        
        Args:
            U, S, V: t-SVD分解结果
            rank: 目标秩
            
        Returns:
            低秩重构的张量
        """
        # 截断奇异值
        S_truncated = S.copy()
        n1, n2, n3 = S.shape
        min_dim = min(n1, n2)
        
        # 确保rank不超过最小维度
        rank = min(rank, min_dim)
        
        # 截断奇异值矩阵
        for i in range(n3):
            # 将rank之后的对角元素设为0
            for j in range(rank, min_dim):
                S_truncated[j, j, i] = 0
    
        # 重构张量
        V_transposed = np.transpose(V, (1, 0, 2))
        reconstructed = self.t_product(self.t_product(U, S_truncated), V_transposed)
        
        return reconstructed
    
    def prepare_multiview_tensor(self, data_views):
        """
        将多视角数据组织成张量形式
        
        Args:
            data_views: 列表，每个元素是一个视角的特征矩阵
            
        Returns:
            三阶张量: [n_samples, n_features, n_views]
        """
        # 确保所有视角的样本数相同
        n_samples = data_views[0].shape[0]
        n_features = data_views[0].shape[1]
        n_views = len(data_views)
        
        # 构建三阶张量
        tensor = np.zeros((n_samples, n_features, n_views))
        for i, view_data in enumerate(data_views):
            # 确保特征维度一致
            if view_data.shape[1] != n_features:
                # 使用线性投影进行维度对齐
                if view_data.shape[1] < n_features:
                    # 填充到目标维度
                    padding = np.zeros((n_samples, n_features - view_data.shape[1]))
                    view_data = np.concatenate([view_data, padding], axis=1)
                else:
                    # 截断到目标维度
                    view_data = view_data[:, :n_features]
            
            tensor[:, :, i] = view_data
        
        return tensor 