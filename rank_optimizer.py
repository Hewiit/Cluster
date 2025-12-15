import numpy as np
from tensor_svd import TensorSVD

class RankOptimizer:
    """基于ADMM的张量低秩重构优化器"""
    
    def __init__(self):
        self.tensor_svd = TensorSVD()
        self.max_iter = 10
        self.tol = 1e-8
        self.rho = 0.5  # ADMM参数
    
    def comprehensive_rank_selection(self, S, data_views, method='admm'):
        """
        使用ADMM方法确定最优秩
        
        Args:
            S: 奇异值张量
            data_views: 多视角数据
            method: 选择方法（仅支持'admm'）
            
        Returns:
            最优秩
        """
        print("=== 使用ADMM方法确定最优秩 ===")
        
        try:
            optimal_rank, _, convergence_info = self.admm_tensor_low_rank_optimization(
                data_views, lambda_param=0.01
            )
            print(f"ADMM优化方法: 秩 = {optimal_rank}")
            
            return optimal_rank
            
        except Exception as e:
            print(f"ADMM方法失败: {str(e)}")
            # 使用简单的秩估计作为备选
            tensor = self.tensor_svd.prepare_multiview_tensor(data_views)
            U, S_svd, V = self.tensor_svd.t_svd_decomposition(tensor)
            optimal_rank = min(S_svd.shape[0], S_svd.shape[1]) // 2
            print(f"使用备选秩: {optimal_rank}")
        return optimal_rank
    
    def admm_tensor_low_rank_optimization(self, data_views, lambda_param=0.01, max_iter=None, tol=None):
        """
        基于ADMM的张量低秩重构优化
        
        优化问题：
        min ||X - L||_F^2 + λ||L||_*
        s.t. L = Z
        
        其中：
        - X是原始张量
        - L是低秩张量
        - Z是辅助变量
        - ||·||_*是核范数（奇异值之和）
        
        Args:
            data_views: 多视角数据列表
            lambda_param: 正则化参数
            max_iter: 最大迭代次数
            tol: 收敛容差
            
        Returns:
            optimal_rank: 最优秩
            L_optimal: 最优低秩张量
            convergence_info: 收敛信息
        """
        if max_iter is None:
            max_iter = self.max_iter
        if tol is None:
            tol = self.tol
            
        print(f"=== ADMM张量低秩重构优化 ===")
        print(f"正则化参数 λ = {lambda_param}")
        print(f"收敛容差 = {tol}")
        
        # 构建原始张量
        X = self.tensor_svd.prepare_multiview_tensor(data_views)
        print(f"原始张量形状: {X.shape}")
        
        # 初始化变量
        n1, n2, n3 = X.shape
        L = X.copy()  # 低秩张量
        Z = X.copy()  # 辅助变量
        U = np.zeros_like(X)  # 拉格朗日乘子
        
        # 记录收敛信息
        primal_residuals = []
        dual_residuals = []
        objective_values = []
        
        print("开始ADMM迭代...")
        
        for iter_num in range(max_iter):
            # 步骤1: 更新L (最小化重构误差 + 正则化项)
            L_old = L.copy()
            
            # 计算Z - U/rho
            temp = Z - U / self.rho
            
            # 使用软阈值操作更新L
            L = self._soft_threshold_update(X, temp, lambda_param, self.rho)
            
            # 步骤2: 更新Z (投影到低秩空间)
            Z_old = Z.copy()
            Z = L + U / self.rho
            
            # 对Z进行低秩投影
            Z = self._low_rank_projection(Z)
            
            # 步骤3: 更新拉格朗日乘子
            U = U + self.rho * (L - Z)
            
            # 计算收敛指标
            primal_residual = np.linalg.norm(L - Z)
            dual_residual = self.rho * np.linalg.norm(Z - Z_old)
            
            primal_residuals.append(primal_residual)
            dual_residuals.append(dual_residual)
            
            # 计算目标函数值
            reconstruction_error = np.linalg.norm(X - L)**2
            nuclear_norm = self._compute_nuclear_norm(L)
            objective_value = reconstruction_error + lambda_param * nuclear_norm
            objective_values.append(objective_value)
            
            # 检查收敛
            if iter_num % 5 == 0:
                print(f"迭代 {iter_num}: 原始残差 = {primal_residual:.6f}, "
                      f"对偶残差 = {dual_residual:.6f}, 目标值 = {objective_value:.6f}, ")
            
            if primal_residual < tol and dual_residual < tol:
                print(f"ADMM收敛于迭代 {iter_num}")
                break
            
            # 如果目标函数值不再显著下降，提前停止
            if iter_num > 100 and len(objective_values) > 20:
                recent_improvement = objective_values[-1] - objective_values[-20]
                if abs(recent_improvement) < tol * 100:
                    print(f"目标函数收敛，提前停止于迭代 {iter_num}")
                    break
        
        # 确定最优秩
        optimal_rank = self._estimate_rank_from_tensor(L)
        
        print(f"最优秩: {optimal_rank}")
        
        convergence_info = {
            'iterations': iter_num + 1,
            'primal_residuals': primal_residuals,
            'dual_residuals': dual_residuals,
            'objective_values': objective_values,
            'converged': primal_residual < tol and dual_residual < tol
        }
        
        return optimal_rank, L, convergence_info
    
    def _soft_threshold_update(self, X, temp, lambda_param, rho):
        """
        软阈值更新操作
        
        Args:
            X: 原始张量
            temp: 临时变量
            lambda_param: 正则化参数
            rho: ADMM参数
            
        Returns:
            更新后的张量
        """
        # 计算权重
        weight = 1.0 / (1.0 + rho)
        
        # 软阈值更新
        L = weight * X + (1 - weight) * temp
        
        return L
    
    def _low_rank_projection(self, Z):
        """
        低秩投影操作
        
        Args:
            Z: 输入张量
            
        Returns:
            投影后的低秩张量
        """
        # 使用t-SVD进行低秩投影
        U, S, V = self.tensor_svd.t_svd_decomposition(Z)
        
        # 保留主要奇异值（自适应确定秩）
        rank = self._adaptive_rank_selection(S)
        
        # 使用tensor_svd中的low_rank_approximation方法
        Z_low_rank = self.tensor_svd.low_rank_approximation(U, S, V, rank)
        
        return Z_low_rank
    
    def _adaptive_rank_selection(self, S):
        """
        自适应秩选择
        
        Args:
            S: 奇异值张量
            
        Returns:
            选择的秩
        """
        # 计算奇异值的能量分布
        n1, n2, n3 = S.shape
        min_dim = min(n1, n2)
        
        # 提取对角奇异值
        singular_values = []
        for i in range(n3):
            for j in range(min_dim):
                singular_values.append(np.abs(S[j, j, i]))
        
        if not singular_values:
            return 1
        
        # 计算总能量
        total_energy = sum(s**2 for s in singular_values)
        
        # 按大小排序奇异值
        singular_values.sort(reverse=True)
        
        # 计算累积能量
        cumulative_energy = 0
        for i, s in enumerate(singular_values):
            cumulative_energy += s**2
            if cumulative_energy >= 0.95 * total_energy:
                rank = i + 1
                break
        else:
            rank = len(singular_values)
        
        # 确保秩在合理范围内
        rank = min(rank, min_dim)
        rank = max(rank, 1)
        
        return rank
    
    def _compute_nuclear_norm(self, tensor):
        """
        计算张量的核范数
        
        Args:
            tensor: 输入张量
            
        Returns:
            核范数值
        """
        # 使用t-SVD计算核范数
        U, S, V = self.tensor_svd.t_svd_decomposition(tensor)
        
        # 核范数是奇异值之和
        # 只计算对角元素的奇异值
        n1, n2, n3 = S.shape
        min_dim = min(n1, n2)
        nuclear_norm = 0
        
        for i in range(n3):
            for j in range(min_dim):
                nuclear_norm += np.abs(S[j, j, i])
        
        return nuclear_norm
    
    def _estimate_rank_from_tensor(self, tensor):
        """
        从张量估计秩
        
        Args:
            tensor: 输入张量
            
        Returns:
            估计的秩
        """
        # 使用t-SVD分解
        U, S, V = self.tensor_svd.t_svd_decomposition(tensor)
        
        # 计算奇异值的能量分布
        n1, n2, n3 = S.shape
        min_dim = min(n1, n2)
        
        # 提取对角奇异值
        singular_values = []
        for i in range(n3):
            for j in range(min_dim):
                singular_values.append(np.abs(S[j, j, i]))
        
        if not singular_values:
            return 1
        
        # 计算总能量
        total_energy = sum(s**2 for s in singular_values)
        
        # 按大小排序奇异值
        singular_values.sort(reverse=True)
        
        # 计算累积能量
        cumulative_energy = 0
        for i, s in enumerate(singular_values):
            cumulative_energy += s**2
            if cumulative_energy >= 0.95 * total_energy:
                rank = i + 1
                break
        else:
            rank = len(singular_values)
        
        # 确保秩在合理范围内
        rank = min(rank, min_dim)
        rank = max(rank, 1)
        
        return rank
    

    
 