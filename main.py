
# from data_loader import load_data
from pickle import FALSE
from data_loader import load_data
from feature_extractor import FeatureExtractor, get_feature_extractor, init_feature_extractor
from clustering import ClusterAnalyzer, extract_main_tag
from tensor_clustering import MultiViewTensorClustering
from utils import (
    save_clustering_result, 
    print_cluster_stats, 
    print_clustering_metrics, 
    save_features, 
    load_features,
    save_multiview_features,
    load_multiview_features,
    export_clusters_to_events,
    compute_prior_distribution,
    apply_prior_based_sampling,
    compute_sample_weights
)
from config import N_CLUSTERS, FEATURES_DIR, DATASET_TYPE, WEIBO_BASE_DIR, ALL_PLATFORM_BASE_DIR
import os
import numpy as np
import torch
import gc

# 直接指定导出事件数据的目录
CLUSTER_EVENTS_DIR = "cluster_events"

def cleanup_memory():
    """清理显存和内存"""
    print("\n清理显存...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.synchronize()  # 同步CUDA操作
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    # 清理Python垃圾回收
    gc.collect()
    print("显存清理完成")

def load_and_extract_features(use_saved=False, method='tensor', data_source_path=None, source_site=None):
    """
    加载数据并提取特征
    
    Args:
        use_saved: 是否使用已保存的特征
        method: 聚类方法 ('traditional' 或 'tensor')
        data_source_path: 数据源路径，如果为None则使用config中的默认路径
        source_site: 针对ALL_PLATFORM数据的来源网站过滤条件
    
    Returns:
        tuple: (features, labels) 其中features根据method不同而不同
    """
    if method not in ['tensor', 'traditional']:
        raise ValueError(f"不支持的特征提取方法: {method}")
    can_use_saved = method in ['tensor', 'traditional']
    if can_use_saved and use_saved and os.path.exists(FEATURES_DIR) and os.listdir(FEATURES_DIR):
        print("尝试加载已保存的特征...")
        
        if method == 'tensor':
            # 尝试加载多视角特征
            multiview_features, labels = load_multiview_features()
            print("多视角特征加载成功")
            return multiview_features, labels
        else:
            # 加载传统特征
            image_features, text_features, tags, ner_features, labels = load_features()
            print("传统特征加载成功")
            return (image_features, text_features, tags, ner_features), labels
            
    else:
        print("开始加载数据并提取新特征...")
        # 加载数据并提取特征
        texts, image_paths, labels = load_data(
            data_source_path=data_source_path,
            source_site=source_site
        )
        
        if method == 'tensor':
            feature_extractor = get_feature_extractor()
            # 提取多视角特征用于t-SVD聚类
            multiview_features = feature_extractor.extract_multiview_features(texts, image_paths)
            # 保存多视角特征
            save_multiview_features(multiview_features, labels)
            return multiview_features, labels
        elif method == 'traditional':
            feature_extractor = get_feature_extractor()
            # 传统特征提取
            image_features, text_features, tags, ner_features = feature_extractor.extract_features(texts, image_paths)
            # 保存新提取的特征
            save_features(image_features, text_features, tags, ner_features, labels)
            return (image_features, text_features, tags, ner_features), labels

def perform_clustering(features, labels, method='tensor', use_prior=True, max_samples_per_event=1000, min_samples_per_event=1):
    """
    执行聚类分析
    
    Args:
        features: 特征数据
        labels: 标签数据
        method: 聚类方法 ('traditional' 或 'tensor')
        use_prior: 是否使用先验分布（默认True）
        max_samples_per_event: 每个事件的最大样本数（用于先验采样）
        min_samples_per_event: 每个事件的最小样本数（用于先验采样）
    
    Returns:
        tuple: (clusters, metrics, additional_info)
    """
    print("进行聚类分析...")
    
    # 计算先验分布
    if use_prior and labels is not None:
        prior_dist = compute_prior_distribution(labels)
        
        # 根据先验分布进行采样
        if method == 'tensor':
            # 对多视角特征进行采样
            sampled_text_features, sampled_labels, sampled_indices = apply_prior_based_sampling(
                features['text'], labels, max_samples_per_event, min_samples_per_event
            )
            sampled_image_features = features['image'][sampled_indices]
            sampled_entity_features = features['entity'][sampled_indices]
            
            # 准备多视角数据
            data_views = [
                sampled_text_features,
                sampled_image_features,
                sampled_entity_features
            ]
            
            # 计算样本权重
            sample_weights = compute_sample_weights(sampled_labels, alpha=0.5)
            
            # 创建t-SVD聚类器
            tensor_clustering = MultiViewTensorClustering(
                n_clusters=None,  
                rank=None,  
                clustering_method='kmeans'
            )
            
            # 执行聚类
            clusters, metrics, optimal_rank = tensor_clustering.fit(data_views, sampled_labels, sample_weights)
            
            print(f"最优张量秩: {optimal_rank}")
            
            # 将采样后的聚类结果映射回原始数据
            # 对于未采样的样本，使用最近邻分配
            from sklearn.neighbors import NearestNeighbors
            if len(sampled_indices) < len(labels):
                # print(f"将聚类结果映射回原始数据 ({len(sampled_indices)} -> {len(labels)})...")
                # 使用采样后的特征和聚类结果训练最近邻分类器
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(sampled_text_features)
                all_indices = set(range(len(labels)))
                unsampled_indices = sorted(list(all_indices - set(sampled_indices)))
                
                if len(unsampled_indices) > 0:
                    unsampled_text_features = features['text'][unsampled_indices]
                    _, nearest_indices = nn.kneighbors(unsampled_text_features)
                    nearest_indices = nearest_indices.flatten()
                    
                    # 创建完整的聚类结果
                    full_clusters = np.zeros(len(labels), dtype=int)
                    for i, orig_idx in enumerate(sampled_indices):
                        full_clusters[orig_idx] = clusters[i]
                    for i, orig_idx in enumerate(unsampled_indices):
                        nearest_sampled_idx = sampled_indices[nearest_indices[i]]
                        nearest_cluster = full_clusters[nearest_sampled_idx]
                        full_clusters[orig_idx] = nearest_cluster
                    
                    clusters = full_clusters
            
            return clusters, metrics, {
                'tensor_clustering': tensor_clustering,
                'optimal_rank': optimal_rank,
                'sampled_indices': sampled_indices,
                'prior_distribution': prior_dist
            }
        else:
            # 传统方法：对特征进行采样
            image_features, text_features, tags, ner_features = features
            # 合并特征用于采样
            if len(image_features.shape) > 2:
                image_features_flat = image_features.reshape(image_features.shape[0], -1)
            else:
                image_features_flat = image_features
            if len(text_features.shape) > 2:
                text_features_flat = text_features.reshape(text_features.shape[0], -1)
            else:
                text_features_flat = text_features
            
            combined_features = np.concatenate([image_features_flat, text_features_flat], axis=1)
            sampled_features, sampled_labels, sampled_indices = apply_prior_based_sampling(
                combined_features, labels, max_samples_per_event, min_samples_per_event
            )
            
            # 分离采样后的特征
            img_dim = image_features_flat.shape[1]
            sampled_image_features = sampled_features[:, :img_dim]
            sampled_text_features = sampled_features[:, img_dim:]
            
            # 恢复原始形状
            if len(image_features.shape) > 2:
                sampled_image_features = sampled_image_features.reshape(-1, *image_features.shape[1:])
            if len(text_features.shape) > 2:
                sampled_text_features = sampled_text_features.reshape(-1, *text_features.shape[1:])
            
            analyzer = ClusterAnalyzer()
            # 预处理与降维
            try:
                reduced_features, _ = analyzer.preprocess_features(sampled_image_features, sampled_text_features)
            except Exception:
                fused_features = np.concatenate([
                    sampled_image_features.reshape(sampled_image_features.shape[0], -1),
                    sampled_text_features.reshape(sampled_text_features.shape[0], -1)
                ], axis=1)
                reduced_features = fused_features

            # 聚类
            clusters, metrics = analyzer.perform_clustering(reduced_features, sampled_labels, N_CLUSTERS)
            
            # 映射回原始数据（如果需要）
            if len(sampled_indices) < len(labels):
                # print(f"将聚类结果映射回原始数据 ({len(sampled_indices)} -> {len(labels)})...")
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(reduced_features)
                
                all_indices = set(range(len(labels)))
                unsampled_indices = sorted(list(all_indices - set(sampled_indices)))
                
                if len(unsampled_indices) > 0:
                    # 为未采样样本提取特征
                    unsampled_img_features = image_features[unsampled_indices]
                    unsampled_text_features = text_features[unsampled_indices]
                    try:
                        unsampled_reduced, _ = analyzer.preprocess_features(unsampled_img_features, unsampled_text_features)
                    except:
                        unsampled_reduced = np.concatenate([
                            unsampled_img_features.reshape(unsampled_img_features.shape[0], -1),
                            unsampled_text_features.reshape(unsampled_text_features.shape[0], -1)
                        ], axis=1)
                    
                    _, nearest_indices = nn.kneighbors(unsampled_reduced)
                    nearest_indices = nearest_indices.flatten()
                    
                    full_clusters = np.zeros(len(labels), dtype=int)
                    for i, orig_idx in enumerate(sampled_indices):
                        full_clusters[orig_idx] = clusters[i]
                    for i, orig_idx in enumerate(unsampled_indices):
                        nearest_sampled_idx = sampled_indices[nearest_indices[i]]
                        nearest_cluster = full_clusters[nearest_sampled_idx]
                        full_clusters[orig_idx] = nearest_cluster
                    
                    clusters = full_clusters

            return clusters, metrics, {
                'reduced_features': reduced_features,
                'sampled_indices': sampled_indices,
                'prior_distribution': prior_dist
            }
    else:
        # 不使用先验分布，使用原始方法
        if method == 'tensor':
            # 使用t-SVD张量聚类
            print("使用t-SVD张量聚类方法...")
            
            # 准备多视角数据
            data_views = [
                features['text'],
                features['image'],
                features['entity']
            ]
            
            # 创建t-SVD聚类器
            tensor_clustering = MultiViewTensorClustering(
                n_clusters=None,  
                rank=None,  
                clustering_method='kmeans'
            )
            
            # 执行聚类
            clusters, metrics, optimal_rank = tensor_clustering.fit(data_views, labels)
            
            print(f"最优张量秩: {optimal_rank}")
            
            return clusters, metrics, {
                'tensor_clustering': tensor_clustering,
                'optimal_rank': optimal_rank
            }
        
        elif method == 'traditional':
            # 传统聚类方法
            print("使用传统聚类方法...")
            analyzer = ClusterAnalyzer()

            image_features, text_features, tags, ner_features = features
            # 预处理与降维（如果预处理使用UMAP不可用，则直接拼接特征）
            try:
                reduced_features, _ = analyzer.preprocess_features(image_features, text_features)
            except Exception:
                fused_features = np.concatenate([
                    image_features,
                    text_features.reshape(text_features.shape[0], -1)
                ], axis=1)
                reduced_features = fused_features

            # 聚类
            clusters, metrics = analyzer.perform_clustering(reduced_features, labels, N_CLUSTERS)

            return clusters, metrics, {
                'reduced_features': reduced_features
            }
        
        else:
            raise ValueError(f"不支持的聚类方法: {method}")

def export_results(clusters, features, method='tensor', data_source_path=None, additional_info=None):
    """
    导出聚类结果
    
    Args:
        clusters: 聚类结果
        features: 特征数据
        method: 聚类方法
    """
    if DATASET_TYPE in ["WEIBO", "ALL_PLATFORM"]:
        print("\n导出聚类结果为事件数据...")
        if method == 'tensor':
            # 使用多视角特征中的NER信息
            ner_features_for_export = features['ner_info']
        elif method == 'traditional':
            # 使用传统特征中的NER信息
            ner_features_for_export = features[3]
        else:
            ner_features_for_export = features.get('ner_info')

        if data_source_path:
            export_base_dir = data_source_path
        else:
            export_base_dir = WEIBO_BASE_DIR if DATASET_TYPE == "WEIBO" else ALL_PLATFORM_BASE_DIR

        export_clusters_to_events(
            clusters,
            ner_features_for_export,
            CLUSTER_EVENTS_DIR,
            export_base_dir
        )

def print_results(clusters, metrics, labels):
    """
    打印聚类结果
    
    Args:
        clusters: 聚类结果
        metrics: 聚类指标
        labels: 标签数据
    """
    # 打印结果
    print(f"\n聚类结果：")
    print_clustering_metrics(metrics)
    
    # 根据数据集类型选择是否显示标签分布
    if DATASET_TYPE in ["CRISIS", "WEIBO", "ALL_PLATFORM"]:
        print_cluster_stats(clusters, labels)
    else:
        print_cluster_stats(clusters)

def forward(data_source_path=None, use_saved=False, method='traditional', min_posts=1, source_site=None, 
            use_prior=True, max_samples_per_event=1200, min_samples_per_event=1):
    """
    总的forward函数，执行完整的聚类流程
    
    Args:
        data_source_path: 源数据路径（目前未使用，保留接口）
        use_saved: 是否使用已保存的特征
        method: 聚类方法 ('traditional' 或 'tensor')
        min_posts: 最小帖子数量（用于数据过滤）
        source_site: 针对ALL_PLATFORM数据的来源网站过滤条件
        use_prior: 是否使用先验分布（默认True）
        max_samples_per_event: 每个事件的最大样本数（用于先验采样，默认1000）
        min_samples_per_event: 每个事件的最小样本数（用于先验采样，默认10）
    
    Returns:
        dict: 包含聚类结果、指标和附加信息的字典
    """
    try:
        # 清理显存
        cleanup_memory()
        
        # 1. 加载数据并提取特征
        features, labels = load_and_extract_features(
            use_saved,
            method,
            data_source_path,
            source_site
        )
        
        # 2. 执行聚类分析（使用先验分布）
        clusters, metrics, additional_info = perform_clustering(
            features, labels, method, 
            use_prior=use_prior,
            max_samples_per_event=max_samples_per_event,
            min_samples_per_event=min_samples_per_event
        )
        
        if clusters is None:
            print("聚类失败")
            return None
        
        # 3. 打印结果
        print_results(clusters, metrics, labels)
        
        # 4. 导出结果
        export_results(clusters, features, method, data_source_path, additional_info)
        
        # 5. 清理显存
        cleanup_memory()
        
        # 返回结果
        result = {
            'clusters': clusters,
            'metrics': metrics,
            'labels': labels,
            'features': features,
            'additional_info': additional_info
        }
        
        print("\n聚类流程完成！")
        return result
        
    except Exception as e:
        print(f"聚类流程出错: {str(e)}")
        cleanup_memory()
        raise

def main():
    """主函数，调用forward函数"""
    # 直接指定参数
    data_source_path = '/mllms/houwenzheng/LLaMA-Factory/Multicluster/event_wh_新浪微博_20250723_20250729'  # 源数据路径（可选，如果为None则使用config中的WEIBO_BASE_DIR）
    use_saved =  False       # 是否使用已保存的特征
    method = 'traditional'        # 聚类方法: 'traditional' 或 'tensor'
    min_posts = 1          # 最小帖子数量（用于数据过滤）
    source_site = "新浪微博"      # ALL_PLATFORM数据的来源网站过滤条件

    # 初始化encoder
    init_feature_extractor()

    # 调用forward函数
    result = forward(
        data_source_path=data_source_path,
        use_saved=use_saved,
        method=method,
        min_posts=min_posts,
        source_site=source_site
    )
    
    if result is None:
        print("聚类失败")
        return 1
    
    return 0

if __name__ == "__main__":
    main()