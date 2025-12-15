import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
# from cuml.cluster import DBSCAN as cuDBSCAN
# from cuml.cluster import KMeans as cuKMeans
# from cuml.cluster import HDBSCAN as cuHDBSCAN
# from cuml.manifold import UMAP as cuUMAP
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from collections import defaultdict, Counter
import warnings
import os
from config import *

class ClusterAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        # 设备配置
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def preprocess_features(self, image_features, text_features):
        """预处理和融合特征"""
        # 标准化特征
        image_features_scaled = self.scaler.fit_transform(image_features)
        text_features_scaled = self.scaler.fit_transform(text_features.reshape(text_features.shape[0], -1))
        
        # 特征维度对齐和拼接
        fused_features = self._feature_fusion(image_features_scaled, text_features_scaled)
        
        # 使用UMAP降维
        print("\n使用UMAP进行降维...")
        print(f"原始特征维度: {fused_features.shape}")
        
        umap = cuUMAP(
            n_components=1024,  # 降到1024维
            n_neighbors=10,     # 邻居数
            min_dist=0.1,       # 最小距离
            metric='euclidean'  # 距离度量方式
        )
        
        reduced_features = umap.fit_transform(fused_features)
        
        return reduced_features, (image_features_scaled, text_features_scaled)

    def _feature_fusion(self, image_features, text_features):
        """特征对齐和拼接
        
        Args:
            image_features: 图像特征 (n_samples, n_image_features)
            text_features: 文本特征 (n_samples, n_text_features)
            
        Returns:
            融合后的特征 (n_samples, n_image_features + n_text_features)
        """
        # 确保特征维度匹配
        if image_features.shape[1] != text_features.shape[1]:
            # 使用线性投影进行特征对齐
            max_dim = max(image_features.shape[1], text_features.shape[1])
            
            if image_features.shape[1] < max_dim:
                # 创建投影矩阵
                projection = np.random.randn(image_features.shape[1], max_dim - image_features.shape[1])
                projection = projection / np.linalg.norm(projection, axis=0, keepdims=True)  # 归一化
                # 投影到更高维度
                additional_features = np.dot(image_features, projection)
                image_features = np.concatenate([image_features, additional_features], axis=1)
            
            if text_features.shape[1] < max_dim:
                # 创建投影矩阵
                projection = np.random.randn(text_features.shape[1], max_dim - text_features.shape[1])
                projection = projection / np.linalg.norm(projection, axis=0, keepdims=True)  # 归一化
                # 投影到更高维度
                additional_features = np.dot(text_features, projection)
                text_features = np.concatenate([text_features, additional_features], axis=1)
        
        # 使用注意力机制进行特征融合
        # 计算注意力权重
        attention_weights = np.expand_dims(np.sum(image_features * text_features, axis=1), axis=1)
        attention_weights = F.softmax(torch.from_numpy(attention_weights), dim=0).numpy()
        
        # 加权融合
        weighted_image = image_features * attention_weights
        weighted_text = text_features * (1 - attention_weights)
        
        # 拼接加权后的特征
        fused_features = np.concatenate([weighted_image, weighted_text], axis=1)
        
        # L2标准化
        fused_features = fused_features / np.linalg.norm(fused_features, axis=1, keepdims=True)
        
        return fused_features

    # Twitter100k聚类方法（已注释）
    # def find_optimal_clusters(self, reduced_features, sample_tags):
    #     """找到最优的聚类数量"""
    #     best_score = float('-inf')
    #     best_clusters = None
    #     best_n_clusters = 0
    #     best_metrics = {}
    # 
    #     # 初始化最大值
    #     max_metrics = self._calculate_max_metrics(reduced_features, sample_tags)
    # 
    #     # 尝试不同的簇数
    #     for n_clusters in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
    #         clusters, metrics = self._evaluate_clustering(reduced_features, n_clusters, sample_tags, max_metrics)
    #         
    #         # 使用三个指标的加权平均作为综合得分
    #         combined_score = (
    #             metrics["纯度"] * 0.4 +
    #             metrics["NMI"] * 0.3 +
    #             metrics["ARI"] * 0.3
    #         )
    #         
    #         if combined_score > best_score:
    #             best_score = combined_score
    #             best_clusters = clusters
    #             best_n_clusters = n_clusters
    #             best_metrics = metrics
    #     
    #     return best_clusters, best_metrics, best_n_clusters

    def auto_clustering(self, reduced_features, true_labels=None):
        """自动确定簇数的聚类方法
        
        Args:
            reduced_features: 降维后的特征
            true_labels: 真实标签（可选）
            
        Returns:
            clusters: 聚类结果
            metrics: 评估指标
        """
        print("执行自动聚类...")
        
        # 确保数据类型和格式正确
        reduced_features = np.asarray(reduced_features, dtype=np.float32)
        if len(reduced_features.shape) == 1:
            reduced_features = reduced_features.reshape(-1, 1)
        
        print(f"特征维度: {reduced_features.shape}")
        
        best_score = float('-inf')
        best_clusters = None
        best_method = None
        best_params = None
        
        # 尝试HDBSCAN
        print("\n尝试HDBSCAN聚类...")
        try:
            hdbscan = cuHDBSCAN(
                min_cluster_size=5,  
                min_samples=2,        
                metric='euclidean',
                cluster_selection_epsilon=0.5,  # 降低epsilon值
                cluster_selection_method='eom',  # 使用leaf方法，更稳定
                alpha=1.0
            )
            clusters = hdbscan.fit_predict(reduced_features)
            
            # 计算评估指标
            score, metrics = self._evaluate_clustering(clusters, reduced_features, true_labels)
            print(f"HDBSCAN聚类得分: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_clusters = clusters
                best_method = "HDBSCAN"
        except Exception as e:
            print(f"HDBSCAN聚类失败: {str(e)}")
        
        # 输出最佳结果
        if best_clusters is not None:
            print(f"\n最佳聚类方法: {best_method}")
            print(f"最佳得分: {best_score:.4f}")
            return best_clusters, self._calculate_metrics(best_clusters, true_labels)
        else:
            print("\n自动聚类失败，回退到谱聚类...")
            return self.perform_clustering(reduced_features, true_labels, N_CLUSTERS)
    
    def _evaluate_clustering(self, clusters, features, true_labels=None):
        """评估聚类结果
        
        Args:
            clusters: 聚类标签
            features: 特征向量
            true_labels: 真实标签（可选）
            
        Returns:
            score: 综合得分
            metrics: 评估指标字典
        """
        metrics = {}
        
        # 确保数据类型正确
        clusters = np.asarray(clusters, dtype=np.int32)
        features = np.asarray(features, dtype=np.float32)
        
        # 排除噪声点
        non_noise_mask = clusters != -1
        if np.sum(non_noise_mask) == 0:
            print("警告：所有点都被标记为噪声点")
            return float('-inf'), metrics
            
        non_noise_clusters = clusters[non_noise_mask]
        non_noise_features = features[non_noise_mask]
        non_noise_labels = true_labels[non_noise_mask] if true_labels is not None else None
        
        # 计算噪声点比例
        noise_ratio = 1 - len(non_noise_clusters) / len(clusters)
        
        # 计算轮廓系数
        try:
            silhouette = silhouette_score(non_noise_features, non_noise_clusters)
        except:
            silhouette = 0
            
        # 计算纯度
        purity = calculate_purity(non_noise_clusters, non_noise_labels) if non_noise_labels is not None else 0
        
        # 计算其他指标
        if non_noise_labels is not None:
            metrics["NMI"] = normalized_mutual_info_score(non_noise_labels, non_noise_clusters)
            metrics["ARI"] = adjusted_rand_score(non_noise_labels, non_noise_clusters)
        else:
            metrics["NMI"] = 0
            metrics["ARI"] = 0
            
        metrics["纯度"] = purity
        metrics["噪声点比例"] = noise_ratio
        
        # 综合得分
        score = silhouette + purity - noise_ratio * 0.5
        
        return score, metrics

    def perform_clustering(self, reduced_features, true_labels, n_clusters):
        """使用谱聚类进行聚类"""
        print("执行谱聚类...")
        
        # 初始化谱聚类模型
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors',  # 使用KNN构建相似度矩阵
            n_neighbors=10,  # KNN的邻居数
            random_state=42
        )
        
        # 执行聚类
        clusters = spectral.fit_predict(reduced_features)
        
        # 计算评估指标
        metrics = {}
        if true_labels is not None and len(true_labels) > 0:
            metrics["纯度"] = calculate_purity(clusters, true_labels)
            metrics["NMI"] = normalized_mutual_info_score(true_labels, clusters)
            metrics["ARI"] = adjusted_rand_score(true_labels, clusters)
        else:
            metrics["纯度"] = 0
            metrics["NMI"] = 0
            metrics["ARI"] = 0
        
        return clusters, metrics
    
    def perform_size_constrained_clustering(self, reduced_features, true_labels, cluster_sizes):
        """使用约束簇大小的聚类方法
        
        Args:
            reduced_features: 降维后的特征
            true_labels: 真实标签（如果有）
            cluster_sizes: 期望的簇大小列表或字典，如 [100, 200, 150] 或 {'事件1': 100, '事件2': 200}
            
        Returns:
            簇标签和评估指标
        """
        print("执行约束簇大小的聚类...")
        
        # 如果输入是字典，转换为列表
        if isinstance(cluster_sizes, dict):
            sizes_list = list(cluster_sizes.values())
            event_names = list(cluster_sizes.keys())
        else:
            sizes_list = cluster_sizes
            event_names = [f"簇{i}" for i in range(len(sizes_list))]
            
        n_clusters = len(sizes_list)
        n_samples = reduced_features.shape[0]
        
        # 先用KMeans获取初始聚类中心
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(reduced_features)
        centroids = kmeans.cluster_centers_
        
        # 计算样本到每个中心的距离
        distances = np.zeros((n_samples, n_clusters))
        for i in range(n_clusters):
            distances[:, i] = np.sum((reduced_features - centroids[i])**2, axis=1)
        
        # 分配样本到簇，考虑簇大小约束
        assigned = np.zeros(n_samples, dtype=bool)
        clusters = np.zeros(n_samples, dtype=int) - 1  # -1表示未分配
        
        # 计算每个簇需要的样本数量
        target_sizes = np.zeros(n_clusters, dtype=int)
        for i, size in enumerate(sizes_list):
            # 等比例缩放目标簇大小，使总和等于样本总数
            target_sizes[i] = int(size / sum(sizes_list) * n_samples)
        
        # 确保所有簇大小之和等于样本总数
        remaining = n_samples - np.sum(target_sizes)
        if remaining > 0:
            # 将剩余样本分配给最大的簇
            largest_cluster = np.argmax(target_sizes)
            target_sizes[largest_cluster] += remaining
            
        print("目标簇大小:")
        for i, (name, size) in enumerate(zip(event_names, target_sizes)):
            print(f"{name}: {size} 个样本")
            
        # 按距离排序分配样本
        for i in range(n_clusters):
            # 获取未分配的样本
            unassigned = np.where(~assigned)[0]
            if len(unassigned) == 0:
                break
                
            # 获取这些样本到当前中心的距离
            cluster_distances = distances[unassigned, i]
            
            # 按距离排序
            sorted_indices = np.argsort(cluster_distances)
            
            # 取前target_sizes[i]个最近的样本
            to_assign = min(target_sizes[i], len(sorted_indices))
            assign_indices = unassigned[sorted_indices[:to_assign]]
            
            # 分配样本
            clusters[assign_indices] = i
            assigned[assign_indices] = True
            
        # 计算最终的簇大小
        final_sizes = Counter(clusters)
        print("\n最终簇大小:")
        for i in range(n_clusters):
            print(f"{event_names[i]}: {final_sizes.get(i, 0)} 个样本")
            
        # 计算评估指标
        metrics = {}
        if true_labels is not None and len(true_labels) > 0:
            metrics["纯度"] = calculate_purity(clusters, true_labels)
            metrics["NMI"] = normalized_mutual_info_score(true_labels, clusters)
            metrics["ARI"] = adjusted_rand_score(true_labels, clusters)
        else:
            metrics["纯度"] = 0
            metrics["NMI"] = 0
            metrics["ARI"] = 0
            
        return clusters, metrics
    
    def estimate_cluster_sizes_from_weibo_data(self):
        """从微博或ALL_PLATFORM数据集的CSV文件估计每个事件的样本数量"""
        if DATASET_TYPE not in ["WEIBO", "ALL_PLATFORM"]:
            print("此方法只适用于微博或多平台数据集")
            return None
            
        # 获取事件列表
        base_dir = WEIBO_BASE_DIR if DATASET_TYPE == "WEIBO" else ALL_PLATFORM_BASE_DIR
        event_dirs = [d for d in os.listdir(base_dir) 
                     if os.path.isdir(os.path.join(base_dir, d))]
        
        # 计算每个事件的数据量
        event_sizes = {}
        for event in event_dirs:
            event_path = os.path.join(base_dir, event)
            csv_files = [f for f in os.listdir(event_path) 
                        if f.endswith('.csv')]
            
            if not csv_files:
                continue
                
            csv_file = next((f for f in csv_files if f.lower().startswith(event.lower())), 
                           csv_files[0])
            csv_path = os.path.join(event_path, csv_file)
                
            try:
                # 统计CSV文件中的行数
                df = pd.read_csv(csv_path, encoding='utf-8')
                event_sizes[event] = len(df)
                # print(f"事件 {event}: {len(df)} 条数据")
            except Exception as e:
                print(f"读取事件 {event} 的CSV文件时出错: {str(e)}")
                continue
        
        return event_sizes

    def _calculate_metrics(self, clusters, true_labels):
        """计算聚类评估指标，排除噪声点（簇-1）"""
        metrics = {}
        if true_labels is not None and len(true_labels) > 0:
            # 排除噪声点
            non_noise_mask = clusters != -1
            non_noise_clusters = clusters[non_noise_mask]
            non_noise_labels = true_labels[non_noise_mask]
            
            if len(non_noise_clusters) > 0:
                metrics["纯度"] = calculate_purity(non_noise_clusters, non_noise_labels)
                metrics["NMI"] = normalized_mutual_info_score(non_noise_labels, non_noise_clusters)
                metrics["ARI"] = adjusted_rand_score(non_noise_labels, non_noise_clusters)
                
                # 添加噪声点比例信息
                noise_ratio = 1 - len(non_noise_clusters) / len(clusters)
                metrics["噪声点比例"] = noise_ratio
                print(f"噪声点比例: {noise_ratio:.2%}")
            else:
                metrics["纯度"] = 0
                metrics["NMI"] = 0
                metrics["ARI"] = 0
                metrics["噪声点比例"] = 1.0
                print("警告：所有点都被标记为噪声点")
        else:
            metrics["纯度"] = 0
            metrics["NMI"] = 0
            metrics["ARI"] = 0
            metrics["噪声点比例"] = 0
        return metrics

def calculate_purity(clusters, true_labels):
    """计算聚类纯度
    
    Args:
        clusters: 预测的聚类标签
        true_labels: 真实标签
    
    Returns:
        float: 聚类纯度分数
    """
    if true_labels is None or len(true_labels) == 0:
        return 0
        
    cluster_label_distribution = defaultdict(Counter)
    
    # 构建聚类结果中每个簇的标签分布
    for cluster_id, label in zip(clusters, true_labels):
        cluster_label_distribution[cluster_id][label] += 1
    
    # 计算正确分类的样本数
    correctly_clustered = 0
    total_samples = len(true_labels)
    
    for cluster_id in cluster_label_distribution:
        # 获取该簇中最多的标签对应的样本数
        max_count = max(cluster_label_distribution[cluster_id].values())
        correctly_clustered += max_count
    
    return correctly_clustered / total_samples

def extract_main_tag(tags):
    """提取主标签（用于Twitter100k数据集）"""
    all_hashtags = [tag for tags_list in tags for tag in tags_list]
    hashtag_counts = Counter(all_hashtags)
    top_20_hashtags = set([tag for tag, _ in hashtag_counts.most_common(20)])

    sample_tags = []
    for tag_list in tags:
        popular_tags = [tag for tag in tag_list if tag in top_20_hashtags]
        if popular_tags:
            most_common_tag = max(popular_tags, key=lambda x: hashtag_counts[x])
            sample_tags.append(most_common_tag)
        else:
            sample_tags.append(None)
    return sample_tags 