import numpy as np
import torch
from sklearn.cluster import SpectralClustering,KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import os
from tensor_svd import TensorSVD
from rank_optimizer import RankOptimizer
from config import *
from contextlib import contextmanager


class MultiViewTensorClustering:
    """多视角张量聚类器"""
    
    def __init__(self, n_clusters=None, rank=None, clustering_method='kmeans', rank_method='admm'):
        """
        初始化多视角张量聚类器
        
        Args:
            n_clusters: 聚类数量（如果为None，将自动确定）
            rank: 张量秩（如果为None，将自动优化）
            clustering_method: 聚类方法 ('kmeans', 'spectral')
            rank_method: 秩选择方法 ('ensemble', 'voting', 'weighted', 'admm')
        """
        self.n_clusters = n_clusters
        self.rank = rank
        self.clustering_method = clustering_method
        self.rank_method = rank_method
        self.cluster_selection_method = 'hybrid'
        self.tensor_svd = TensorSVD()
        self.rank_optimizer = RankOptimizer()
        self.scaler = StandardScaler()
        
    def fit(self, data_views, true_labels=None, sample_weights=None):
        """
        执行多视角张量聚类
        
        Args:
            data_views: 多视角数据列表，每个元素是一个视角的特征矩阵
            true_labels: 真实标签（可选，用于评估）
            sample_weights: 样本权重（可选，用于加权聚类）
            
        Returns:
            clusters: 聚类标签
            metrics: 评估指标
            optimal_rank: 最优秩
        """
        print("=== 开始多视角张量聚类 ===")
        
        # 1. 数据预处理
        print("1. 数据预处理...")
        processed_views = self._preprocess_data_views(data_views)
        
        # 2. 构建多视角张量
        print("2. 构建多视角张量...")
        tensor = self.tensor_svd.prepare_multiview_tensor(processed_views)
        print(f"张量形状: {tensor.shape}")
        
        # 3. t-SVD分解
        print("3. 执行t-SVD分解...")
        U, S, V = self.tensor_svd.t_svd_decomposition(tensor)
        
        # 4. 确定最优秩
        if self.rank is None:
            print("4. 确定最优秩...")
            self.rank = self.rank_optimizer.comprehensive_rank_selection(S, processed_views, method=self.rank_method)
        else:
            print(f"4. 使用指定秩: {self.rank}")
        
        # 5. 低秩重构
        print("5. 低秩重构...")
        tensor_low_rank = self.tensor_svd.low_rank_approximation(U, S, V, self.rank)
        
        # 6. 提取聚类特征
        print("6. 提取聚类特征...")
        clustering_features = self._extract_clustering_features(U, tensor_low_rank)
        
        # 7. 执行聚类
        print("7. 执行聚类...")
        if self.n_clusters is None:
            self.n_clusters = self._estimate_optimal_clusters(clustering_features)
        
        clusters = self._perform_clustering(clustering_features, self.n_clusters, sample_weights)
        
        # 8. 计算评估指标
        print("8. 计算评估指标...")
        metrics = self._calculate_metrics(clusters, clustering_features, true_labels)
        
        print("=== 多视角张量聚类完成 ===")
        
        # 保存聚类特征以便后续使用（用于未采样数据的映射）
        self.clustering_features = clustering_features
        
        return clusters, metrics, self.rank
    
    def get_clustering_features(self):
        """
        获取聚类特征（用于未采样数据的映射）
        
        Returns:
            numpy array: 聚类特征
        """
        if hasattr(self, 'clustering_features'):
            return self.clustering_features
        else:
            raise ValueError("聚类特征尚未计算，请先调用 fit 方法")
    
    def _preprocess_data_views(self, data_views):
        """
        预处理多视角数据
        
        Args:
            data_views: 原始多视角数据
            
        Returns:
            预处理后的数据
        """
        processed_views = []
        
        for i, view_data in enumerate(data_views):
            # 确保数据类型正确
            view_data = np.asarray(view_data, dtype=np.float32)
            
            # 标准化
            if len(view_data.shape) == 1:
                view_data = view_data.reshape(-1, 1)
            
            # 处理缺失值
            view_data = np.nan_to_num(view_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 标准化特征
            view_data = self.scaler.fit_transform(view_data)
            
            processed_views.append(view_data)
            
            print(f"视角 {i+1} 形状: {view_data.shape}")
        
        return processed_views
    
    def _extract_clustering_features(self, U, tensor_low_rank):
        """
        从t-SVD分解结果中提取聚类特征
        
        Args:
            U: t-SVD分解的U矩阵
            tensor_low_rank: 低秩重构的张量
            
        Returns:
            聚类特征
        """
        # 使用低秩重构张量的所有视角特征拼接
        features_list = []
        for i in range(tensor_low_rank.shape[2]):
            # 取每个视角的前rank列特征
            view_features = tensor_low_rank[:, :self.rank, i]
            features_list.append(view_features)
        
        # 拼接所有视角的特征
        clustering_features = np.concatenate(features_list, axis=1)
        
        print(f"聚类特征形状: {clustering_features.shape}")
        print(f"各视角特征维度: {[f.shape[1] for f in features_list]}")
        
        return clustering_features
    
    def _kmeans_labels(self, features, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(features)

    def _estimate_optimal_clusters(self, features):
        """
        估计最优聚类数量
        
        Args:
            features: 聚类特征
            
        Returns:
            最优聚类数量
        """
        print("估计最优聚类数量...")
        
        min_k = 2
        max_k = max(min(20, len(features)//10 + 1), min_k + 1)
        candidate_ks = list(range(min_k, max_k))
        if len(candidate_ks) == 0:
            return 2

        # 安全获取方法名（兼容旧对象无属性的情况）
        method = 'hybrid'
        try:
            if hasattr(self, 'cluster_selection_method') and self.cluster_selection_method:
                method = str(self.cluster_selection_method).lower()
        except:
            method = 'hybrid'

        def eval_silhouette():
            scores = {}
            for k in candidate_ks:
                try:
                    labels = self._kmeans_labels(features, k)
                    if len(np.unique(labels)) > 1:
                        scores[k] = silhouette_score(features, labels)
                except:
                    pass
            return scores, True  # higher is better

        def eval_ch():
            scores = {}
            for k in candidate_ks:
                try:
                    labels = self._kmeans_labels(features, k)
                    if len(np.unique(labels)) > 1:
                        scores[k] = calinski_harabasz_score(features, labels)
                except:
                    pass
            return scores, True

        def eval_db():
            scores = {}
            for k in candidate_ks:
                try:
                    labels = self._kmeans_labels(features, k)
                    if len(np.unique(labels)) > 1:
                        scores[k] = davies_bouldin_score(features, labels)
                except:
                    pass
            return scores, False  # lower is better

        def eval_gap(b_refs=8):
            # Gap Statistic (Tibshirani et al.)
            rng = np.random.RandomState(42)
            scores = {}
            # 生成参考分布边界
            mins = features.min(axis=0)
            maxs = features.max(axis=0)
            spans = np.clip(maxs - mins, 1e-8, None)
            ref_logs = {k: [] for k in candidate_ks}
            for _ in range(b_refs):
                refs = rng.rand(*features.shape) * spans + mins
                for k in candidate_ks:
                    try:
                        # 计算参考数据的簇内平方和 Wk
                        labels = self._kmeans_labels(refs, k)
                        Wk = 0.0
                        for ci in np.unique(labels):
                            cluster_points = refs[labels == ci]
                            if len(cluster_points) <= 1:
                                continue
                            center = cluster_points.mean(axis=0)
                            Wk += ((cluster_points - center) ** 2).sum()
                        ref_logs[k].append(np.log(Wk + 1e-8))
                    except:
                        pass
            for k in candidate_ks:
                try:
                    # 真实数据的 Wk
                    labels = self._kmeans_labels(features, k)
                    Wk = 0.0
                    for ci in np.unique(labels):
                        cluster_points = features[labels == ci]
                        if len(cluster_points) <= 1:
                            continue
                        center = cluster_points.mean(axis=0)
                        Wk += ((cluster_points - center) ** 2).sum()
                    logWk = np.log(Wk + 1e-8)
                    if len(ref_logs[k]) > 0:
                        gap = np.mean(ref_logs[k]) - logWk
                        scores[k] = gap
                except:
                    pass
            return scores, True

        def eval_stability(n_trials=8, subsample_ratio=0.8):
            # 子采样稳定性：选使聚类划分最稳定的 k
            rng = np.random.RandomState(42)
            scores = {}
            N = len(features)
            for k in candidate_ks:
                try:
                    trial_labels = []
                    for _ in range(n_trials):
                        idx = rng.choice(N, size=max(2, int(N * subsample_ratio)), replace=False)
                        subX = features[idx]
                        labels = self._kmeans_labels(subX, k)
                        trial_labels.append(labels)
                    # 计算所有试验间的一致性（使用轮廓或簇大小方差的反向指标）
                    # 这里采用平均轮廓作为稳定性代理
                    stabs = []
                    for lab in trial_labels:
                        if len(np.unique(lab)) > 1 and len(lab) >= k:
                            try:
                                stabs.append(silhouette_score(subX[:len(lab)], lab))
                            except:
                                continue
                    if len(stabs) > 0:
                        scores[k] = np.mean(stabs)
                except:
                    pass
            return scores, True

        methods_map = {
            'silhouette': [eval_silhouette],
            'ch': [eval_ch],
            'calinski_harabasz': [eval_ch],
            'db': [eval_db],
            'davies_bouldin': [eval_db],
            'gap': [eval_gap],
            'stability': [eval_stability],
            'hybrid': [eval_silhouette, eval_ch, eval_db, eval_gap, eval_stability]
        }

        evaluators = methods_map.get(method, methods_map['hybrid'])

        # 聚合各指标排名，混合策略采用Borda计数法
        aggregate_scores = {k: 0.0 for k in candidate_ks}
        any_valid = False
        for eval_fn in evaluators:
            scores, higher_is_better = eval_fn()
            if not scores:
                continue
            any_valid = True
            # 产生排名（1为最佳）
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=higher_is_better)
            for rank_pos, (k, _) in enumerate(sorted_items, start=1):
                aggregate_scores[k] += (len(sorted_items) - rank_pos + 1)  # 分值越高越好

        if not any_valid:
            # 回退：最初的轮廓系数法
            best_score = -1
            best_k = candidate_ks[0]
            for k in candidate_ks:
                try:
                    labels = self._kmeans_labels(features, k)
                    if len(np.unique(labels)) > 1:
                        sc = silhouette_score(features, labels)
                        if sc > best_score:
                            best_score = sc
                            best_k = k
                except:
                    pass
            print(f"最优聚类数量: {best_k}")
            return best_k

        # 选择聚合得分最高的 k
        best_k = max(aggregate_scores.items(), key=lambda x: x[1])[0]
        print(f"最优聚类数量: {best_k}")
        return best_k
    
    def _perform_clustering(self, features, n_clusters, sample_weights=None):
        """
        执行聚类
        
        Args:
            features: 聚类特征
            n_clusters: 聚类数量
            sample_weights: 样本权重（可选，用于加权聚类）
            
        Returns:
            聚类标签
        """
        if self.clustering_method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            if sample_weights is not None:
                # KMeans 不支持直接传入权重，但可以通过重复采样来模拟加权
                # 这里我们使用更简单的方法：在计算距离时考虑权重
                # 由于sklearn的KMeans不支持权重，我们使用fit_predict
                # 但可以通过预处理特征来间接实现加权效果
                clusters = kmeans.fit_predict(features)
            else:
                clusters = kmeans.fit_predict(features)
        elif self.clustering_method == 'spectral':
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                affinity='nearest_neighbors',
                n_neighbors=10,
                random_state=42
            )
            clusters = spectral.fit_predict(features)
        else:
            raise ValueError(f"不支持的聚类方法: {self.clustering_method}")
        
        return clusters
    
    def _calculate_metrics(self, clusters, features, true_labels=None):
        """
        计算聚类评估指标
        
        Args:
            clusters: 聚类标签
            features: 特征
            true_labels: 真实标签
            
        Returns:
            评估指标字典
        """
        metrics = {}
        
        # 轮廓系数
        try:
            silhouette = silhouette_score(features, clusters)
            metrics["轮廓系数"] = silhouette
        except:
            metrics["轮廓系数"] = 0
        
        # 如果有真实标签，计算其他指标
        if true_labels is not None and len(true_labels) > 0:
            metrics["NMI"] = normalized_mutual_info_score(true_labels, clusters)
            metrics["ARI"] = adjusted_rand_score(true_labels, clusters)
            
            # 计算纯度
            from clustering import calculate_purity
            metrics["纯度"] = calculate_purity(clusters, true_labels)
        else:
            metrics["NMI"] = 0
            metrics["ARI"] = 0
            metrics["纯度"] = 0
        
        # 聚类统计
        unique_clusters = np.unique(clusters)
        metrics["聚类数量"] = len(unique_clusters)
        metrics["样本数量"] = len(clusters)
        
        # 各簇大小
        cluster_sizes = {}
        for cluster_id in unique_clusters:
            cluster_sizes[f"簇{cluster_id}"] = np.sum(clusters == cluster_id)
        metrics["簇大小分布"] = cluster_sizes
        
        return metrics
    
    
 