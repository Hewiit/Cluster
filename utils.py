import os
import json
import numpy as np
from collections import Counter
from config import *
import pandas as pd
from collections import defaultdict
import glob
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import random
import shutil

def save_clustering_result(clusters, features, ner_features, sample_tags, run_id):
    """保存聚类结果"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 将簇标签和特征进行分组
    clustered_features = {}
    for feature, cluster_label in zip(features, clusters):
        if cluster_label not in clustered_features:
            clustered_features[cluster_label] = []
        clustered_features[cluster_label].append(feature.tolist())
    
    # 转换为列表格式
    clustered_features_list = [
        clustered_features[i] 
        for i in sorted(clustered_features.keys())
    ]
    
    # 保存结果
    result = {
        "clusters": clustered_features_list,
        "features": features.tolist(),
        "sample_tags": sample_tags,
        "ner_features": ner_features
    }
    
    output_file = os.path.join(OUTPUT_DIR, f"clustering_result_{run_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"聚类结果已保存至: {output_file}")

def print_cluster_stats(clusters, true_labels=None):
    """打印聚类统计信息
    
    Args:
        clusters: 聚类结果
        true_labels: 可选，真实标签（事件名称）
    """
    print("\n各个簇下的样本点数量：")
    cluster_counts = Counter(clusters)
    for cluster_id, count in sorted(cluster_counts.items()):
        print(f"簇 {cluster_id}: {count} 个样本点")
        
    # 如果有真实标签，还可以分析每个簇内的标签分布
    if true_labels is not None:
        print("\n各个簇内的事件分布：")
        for cluster_id in sorted(set(clusters)):
            # 获取该簇内的所有样本索引
            indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            
            # 统计标签分布
            label_counter = Counter([true_labels[i] for i in indices])
            total = sum(label_counter.values())
            
            # 打印分布情况
            # print(f"\n簇 {cluster_id} 中的事件分布 (共{total}个样本):")
            for label, count in label_counter.most_common():
                percentage = count / total * 100
                # print(f"  - {label}: {count} 个样本 ({percentage:.1f}%)")
                
            # 计算该簇的纯度
            purity = max(label_counter.values()) / total if total > 0 else 0
            print(f"  簇 {cluster_id} 的纯度: {purity:.4f}")
            
    print("\n" + "-" * 50)

def print_clustering_metrics(metrics):
    """打印聚类评估指标"""
    print("\n聚类评估指标:")
    print("-" * 30)
    print(f"纯度: {metrics['纯度']:.4f}")
    print(f"NMI: {metrics['NMI']:.4f}")
    print(f"ARI: {metrics['ARI']:.4f}")
    print("-" * 30)

def save_features(image_features, text_features, tags, ner_features, true_labels=None):
    """保存提取的特征到文件"""
    # 创建特征保存目录
    if not os.path.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR)
    
    # 保存特征
    np.save(IMAGE_FEATURES_FILE, image_features)
    np.save(TEXT_FEATURES_FILE, text_features)
    
    # 保存标签数据
    if true_labels is not None:
        np.save(TRUE_LABELS_FILE, true_labels)
    
    # 保存tags和NER特征
    with open(TAGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(tags, f, ensure_ascii=False, indent=4)
        
    with open(NER_FEATURES_FILE, 'w', encoding='utf-8') as f:
        json.dump(ner_features, f, ensure_ascii=False, indent=4)
        
    print(f"特征已保存至目录: {FEATURES_DIR}")

def load_features():
    """从文件加载特征"""
    # 检查特征文件是否存在
    if not all(os.path.exists(f) for f in [IMAGE_FEATURES_FILE, TEXT_FEATURES_FILE]):
        raise FileNotFoundError("特征文件不存在，请先提取特征")
    
    # 加载特征
    image_features = np.load(IMAGE_FEATURES_FILE)
    text_features = np.load(TEXT_FEATURES_FILE)
    
    # 加载标签数据（如果存在）
    true_labels = None
    if os.path.exists(TRUE_LABELS_FILE):
        true_labels = np.load(TRUE_LABELS_FILE)
    
    # 加载tags和NER特征
    with open(TAGS_FILE, 'r', encoding='utf-8') as f:
        tags = json.load(f)
        
    with open(NER_FEATURES_FILE, 'r', encoding='utf-8') as f:
        ner_features = json.load(f)
        
    print("已成功加载特征")
    return image_features, text_features, tags, ner_features, true_labels

def save_multiview_features(multiview_features, true_labels=None):
    """保存多视角特征到文件"""
    # 创建特征保存目录
    if not os.path.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR)
    
    # 保存多视角特征
    np.save(os.path.join(FEATURES_DIR, "multiview_text_features.npy"), multiview_features['text'])
    np.save(os.path.join(FEATURES_DIR, "multiview_image_features.npy"), multiview_features['image'])
    np.save(os.path.join(FEATURES_DIR, "multiview_entity_features.npy"), multiview_features['entity'])
    
    # 保存标签数据
    if true_labels is not None:
        np.save(TRUE_LABELS_FILE, true_labels)
    
    # 保存tags和NER特征
    with open(TAGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(multiview_features['tags'], f, ensure_ascii=False, indent=4)
        
    with open(NER_FEATURES_FILE, 'w', encoding='utf-8') as f:
        json.dump(multiview_features['ner_info'], f, ensure_ascii=False, indent=4)
        
    print(f"多视角特征已保存至目录: {FEATURES_DIR}")

def load_multiview_features():
    """从文件加载多视角特征"""
    # 检查多视角特征文件是否存在
    multiview_text_file = os.path.join(FEATURES_DIR, "multiview_text_features.npy")
    multiview_image_file = os.path.join(FEATURES_DIR, "multiview_image_features.npy")
    multiview_entity_file = os.path.join(FEATURES_DIR, "multiview_entity_features.npy")
    
    if not all(os.path.exists(f) for f in [multiview_text_file, multiview_image_file, multiview_entity_file]):
        raise FileNotFoundError("多视角特征文件不存在，请先提取多视角特征")
    
    # 加载多视角特征
    text_features = np.load(multiview_text_file)
    image_features = np.load(multiview_image_file)
    entity_features = np.load(multiview_entity_file)
    
    # 加载标签数据（如果存在）
    true_labels = None
    if os.path.exists(TRUE_LABELS_FILE):
        true_labels = np.load(TRUE_LABELS_FILE)
    
    # 加载tags和NER特征
    with open(TAGS_FILE, 'r', encoding='utf-8') as f:
        tags = json.load(f)
        
    with open(NER_FEATURES_FILE, 'r', encoding='utf-8') as f:
        ner_features = json.load(f)
    
    # 构建多视角特征字典
    multiview_features = {
        'text': text_features,
        'image': image_features,
        'entity': entity_features,
        'tags': tags,
        'ner_info': ner_features
    }
        
    print("已成功加载多视角特征")
    return multiview_features, true_labels

def save_clusters_as_events(clusters, all_data, output_dir, text_indices=None, ner_features=None):
    """
    将聚类结果以事件形式保存，每个簇作为一个事件，并生成对应的目录结构和CSV文件
    
    Args:
        clusters: 聚类结果标签列表
        all_data: 包含所有微博数据的DataFrame
        output_dir: 输出目录
        text_indices: 可选，聚类样本与原始数据的索引映射，如果为None则假设它们顺序一致
    """
    print(f"\n将聚类结果保存为事件数据...")
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 按簇分组数据
    clustered_data = defaultdict(list)
    
    if text_indices is not None:
        # 使用提供的索引映射：聚类结果索引 -> 原始DataFrame索引
        for i, cluster_id in enumerate(clusters):
            if i < len(text_indices):
                orig_idx = text_indices[i]
                if orig_idx < len(all_data):
                    clustered_data[int(cluster_id)].append(orig_idx)
    else:
        # 假设聚类结果和原始数据顺序一致
        for i, cluster_id in enumerate(clusters):
            if i < len(all_data):
                clustered_data[int(cluster_id)].append(i)
    
    # 记录处理的微博数量
    processed_count = 0
    
    # 记录已使用的目录名和对应的数据，用于合并重复事件
    event_data_map = {}
    
    # 用于存储合并后的簇标签和真实标签
    merged_clusters = []
    merged_true_labels = []
    
    # 为每个簇创建一个目录和CSV文件
    for cluster_id, indices in clustered_data.items():
        try:
            # 提取该簇的数据
            cluster_data = all_data.iloc[indices].copy()
            
            # 跳过空的簇
            if len(cluster_data) == 0:
                print(f"跳过空簇 {cluster_id}")
                continue
                
            # 添加簇ID信息
            cluster_data['簇ID'] = cluster_id
            
            event_name = f"事件_{cluster_id}"  # 默认名称
            orig_events_count = None
            if '原事件' in cluster_data.columns:
                # 统计原事件出现频率
                orig_events_count = cluster_data['原事件'].value_counts()
                if not orig_events_count.empty:
                    # 使用出现频率最高的原始事件名称
                    most_common_event = orig_events_count.index[0]
                    event_name = most_common_event
            
            # 创建事件目录名（处理特殊字符）
            safe_event_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in event_name)
            if safe_event_name == "":
                safe_event_name = f"事件_{cluster_id}"
            
            # 添加事件名称列
            cluster_data['事件名称'] = event_name
            
            # 处理所有文本列以避免编码问题
            for col in cluster_data.columns:
                if cluster_data[col].dtype == 'object':
                    try:
                        cluster_data[col] = cluster_data[col].astype(str).apply(
                            lambda x: x.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                        )
                    except Exception as col_error:
                        print(f"处理列 {col} 时出错: {str(col_error)}")
            
            # 检查是否已存在同名事件
            if safe_event_name in event_data_map:
                # 合并到已存在的事件中
                print(f"发现重复事件名称 '{safe_event_name}'，将簇 {cluster_id} 的数据合并到已存在的事件中")
                existing_data = event_data_map[safe_event_name]
                # 合并数据
                merged_data = pd.concat([existing_data, cluster_data], ignore_index=True)
                event_data_map[safe_event_name] = merged_data
                
                # 更新CSV文件
                cluster_dir_name = safe_event_name
                cluster_dir = os.path.join(output_dir, cluster_dir_name)
                csv_filename = f"{safe_event_name}.csv"
                csv_path = os.path.join(cluster_dir, csv_filename)
                
                # 保存合并后的数据
                try:
                    merged_data.to_csv(csv_path, index=False, encoding='utf-8-sig')
                except Exception as csv_error:
                    print(f"保存CSV文件时出错: {str(csv_error)}")
                    # 备用方案：尝试使用其他编码
                    print("尝试使用其他编码保存...")
                    try:
                        merged_data.to_csv(csv_path, index=False, encoding='gbk')
                        print(f"使用GBK编码保存成功")
                    except:
                        # 最后的备用方案：尝试移除所有可能有问题的列
                        print("尝试移除可能包含编码问题的列...")
                        safe_cols = [col for col in merged_data.columns if col in ['簇ID', '事件名称', '原事件', 'id']]
                        if safe_cols:
                            merged_data[safe_cols].to_csv(csv_path, index=False, encoding='utf-8')
                            print(f"保存了部分列到CSV文件")
                
                # 输出合并信息
                print(f"簇 {cluster_id} 已合并到 {safe_event_name}，新增 {len(cluster_data)} 条微博，总计 {len(merged_data)} 条微博")
            else:
                # 新事件，创建新目录和文件
                cluster_dir_name = safe_event_name
                cluster_dir = os.path.join(output_dir, cluster_dir_name)
                
                # 创建目录
                if not os.path.exists(cluster_dir):
                    os.makedirs(cluster_dir)
                
                # 保存簇数据为CSV
                csv_filename = f"{safe_event_name}.csv"
                csv_path = os.path.join(cluster_dir, csv_filename)
                
                # 保存为CSV，使用utf-8-sig编码（带BOM头，Windows更友好）
                try:
                    cluster_data.to_csv(csv_path, index=False, encoding='utf-8-sig')
                except Exception as csv_error:
                    print(f"保存CSV文件时出错: {str(csv_error)}")
                    # 备用方案：尝试使用其他编码
                    print("尝试使用其他编码保存...")
                    try:
                        cluster_data.to_csv(csv_path, index=False, encoding='gbk')
                        print(f"使用GBK编码保存成功")
                    except:
                        # 最后的备用方案：尝试移除所有可能有问题的列
                        print("尝试移除可能包含编码问题的列...")
                        safe_cols = [col for col in cluster_data.columns if col in ['簇ID', '事件名称', '原事件', 'id']]
                        if safe_cols:
                            cluster_data[safe_cols].to_csv(csv_path, index=False, encoding='utf-8')
                            print(f"保存了部分列到CSV文件")
                
                # 记录已使用的事件名和数据
                event_data_map[safe_event_name] = cluster_data
                
                # 输出簇信息
                print(f"簇 {cluster_id} -> {safe_event_name}，包含 {len(cluster_data)} 条微博")

            # 复制该簇对应的图片到事件目录下的images子目录（结构参考原始结果文件）
            if ner_features is not None:
                images_output_dir = os.path.join(cluster_dir, WEIBO_IMAGE_DIRNAME)
                if not os.path.exists(images_output_dir):
                    os.makedirs(images_output_dir)

                copied_images = 0
                # indices 是原始 DataFrame 中的行索引，假设与 ner_features 的顺序对齐
                if isinstance(ner_features, list):
                    for orig_idx in indices:
                        if orig_idx >= len(ner_features):
                            continue
                        info = ner_features[orig_idx]
                        if not isinstance(info, dict):
                            continue
                        image_paths = info.get("image_paths", [])
                        if not image_paths:
                            continue
                        for src_path in image_paths:
                            if not src_path:
                                continue
                            try:
                                if not os.path.exists(src_path):
                                    continue
                                dst_path = os.path.join(images_output_dir, os.path.basename(src_path))
                                # 避免重复拷贝同名文件
                                if not os.path.exists(dst_path):
                                    shutil.copy2(src_path, dst_path)
                                    copied_images += 1
                            except Exception as img_err:
                                print(f"复制图片 {src_path} 时出错: {str(img_err)}")

                if copied_images > 0:
                    print(f"簇 {cluster_id} ({safe_event_name}) 关联图片已复制 {copied_images} 张到 {images_output_dir}")
            
            # 如果有原事件统计，也输出一下
            if orig_events_count is not None and len(orig_events_count) > 0:
                print(f"  - 原事件组成: {dict(orig_events_count[:3])}")
            
            # 更新处理计数
            processed_count += len(cluster_data)
                
        except Exception as e:
            print(f"处理簇 {cluster_id} 时出错: {str(e)}")
    
    # 输出总结信息
    merged_count = sum(1 for name, data in event_data_map.items() if len(data) > 0)
    print(f"\n聚类结果已保存为事件数据，共 {merged_count} 个事件，处理了 {processed_count} 条微博")
    print(f"输出目录: {output_dir}")
    
    return clustered_data

def export_clusters_to_events(clusters, ner_features, output_dir, base_dir, target_date=None):
    """
    将聚类结果导出为事件数据的主函数，封装了加载原始数据和导出聚类结果的全过程
    
    Args:
        clusters: 聚类结果标签列表
        ner_features: 特征提取中的NER信息
        output_dir: 导出事件数据的目录
        base_dir: 原始微博数据的基础目录
        target_date: 目标日期，如果指定则只加载该日期的数据
    
    Returns:
        bool: 导出是否成功
    """
    print("\n准备将聚类结果导出为事件数据...")
    
    # 过滤掉噪声点（标签为-1的点）
    valid_indices = [i for i, label in enumerate(clusters) if label != -1]
    if len(valid_indices) < len(clusters):
        print(f"过滤掉 {len(clusters) - len(valid_indices)} 个噪声点")
        clusters = clusters[valid_indices]
        if ner_features and isinstance(ner_features, list):
            ner_features = [ner_features[i] for i in valid_indices]
    
    # 加载微博数据到一个大DataFrame
    all_weibo_data = []
    event_dirs = [d for d in os.listdir(base_dir) 
                 if os.path.isdir(os.path.join(base_dir, d))]
    
    # 收集所有微博数据和索引映射
    raw_data_indices = {}  # 存储原始数据ID到DataFrame索引的映射
    current_index = 0
    
    for event in event_dirs:
        event_path = os.path.join(base_dir, event)
        
        # 根据是否指定日期来选择CSV文件
        if target_date:
            # 如果指定了日期，查找{event}_{date}.csv文件
            csv_file = f"{event}_{target_date}.csv"
            csv_path = os.path.join(event_path, csv_file)
            
            # 检查文件是否存在
            if not os.path.exists(csv_path):
                print(f"跳过事件 {event}：未找到日期 {target_date} 的数据文件 {csv_file}")
                continue
        else:
            # 如果没有指定日期，使用原来的逻辑
            csv_files = glob.glob(os.path.join(event_path, "*.csv"))
            if not csv_files:
                print(f"警告：在事件 {event} 中未找到CSV文件")
                continue
            csv_file = csv_files[0]  # 使用第一个CSV文件
            csv_path = csv_file
        
        try:
            print(f"正在处理事件 {event} 的数据文件: {csv_file}")
            # 尝试多种编码方式读取CSV文件
            try:
                # 首先尝试UTF-8编码
                df = pd.read_csv(csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    # 如果UTF-8失败，尝试GBK编码（常用于中文Windows系统）
                    df = pd.read_csv(csv_path, encoding='gbk')
                    print(f"使用GBK编码成功读取: {csv_path}")
                except:
                    # 最后尝试通用编码方式
                    df = pd.read_csv(csv_path, encoding='latin1')
                    print(f"使用latin1编码成功读取: {csv_path}")
            
            # 确保所有文本列编码一致
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).apply(
                        lambda x: x.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                    )
            
            df['原事件'] = event  # 添加原事件信息
            
            # 根据ID列创建索引映射
            if 'id' in df.columns:
                for i, row_id in enumerate(df['id']):
                    raw_data_indices[str(row_id)] = current_index + i
            
            current_index += len(df)
            all_weibo_data.append(df)
        except Exception as e:
            print(f"加载CSV文件 {csv_path} 时出错: {str(e)}")
    
    # 合并所有微博数据
    if not all_weibo_data:
        print("未能加载任何微博数据，无法导出事件")
        return False
    
    all_df = pd.concat(all_weibo_data, ignore_index=True)
    print(f"共加载 {len(all_df)} 条微博数据")
    
    # 检查数据量是否与聚类结果匹配
    if len(clusters) != len(all_df):
        print(f"警告: 聚类结果数量 ({len(clusters)}) 与加载的数据条数 ({len(all_df)}) 不匹配!")
        print("将使用最小长度进行处理...")
        min_len = min(len(clusters), len(all_df))
        clusters = clusters[:min_len]
        all_df = all_df.iloc[:min_len].copy()
        # 保持 ner_features 与聚类结果、原始数据长度一致
        if ner_features and isinstance(ner_features, list):
            if len(ner_features) >= min_len:
                ner_features = ner_features[:min_len]
    
    # 如果ner_features中有原始ID信息，构建索引映射
    text_to_data_mapping = []
    if ner_features and isinstance(ner_features, list):
        for i, feat in enumerate(ner_features):
            if isinstance(feat, dict) and 'text' in feat:
                # 尝试从特征信息中提取原始ID
                text = feat.get('text', '')
                # 映射到相应的索引，如果找不到则使用i
                text_to_data_mapping.append(i)
    
    # 如果没有获取到有效的映射，使用None让函数自动处理
    indices_mapping = None
    if len(text_to_data_mapping) > 0:
        indices_mapping = text_to_data_mapping
        print(f"成功构建特征到原始数据的索引映射，共 {len(indices_mapping)} 条")
    
    # 导出聚类结果为事件数据（同时利用 ner_features 中的图片路径）
    result = save_clusters_as_events(clusters, all_df, output_dir, indices_mapping, ner_features)
    return result is not None

def compute_prior_distribution(labels):
    """
    计算基于原始事件标签的先验分布
    
    Args:
        labels: 原始事件标签列表
    
    Returns:
        dict: {事件名: 样本数量}
    """
    event_counts = Counter(labels)
    total_samples = len(labels)
    
    # print(f"\n=== 先验分布统计 ===")
    # print(f"总样本数: {total_samples}")
    # print(f"事件数量: {len(event_counts)}")
    
    # 按样本数量排序
    sorted_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)
    
    return dict(event_counts)

def apply_prior_based_sampling(features, labels, max_samples_per_event=2000, min_samples_per_event=11, random_seed=42):
    """
    基于先验分布进行采样，对大数据集降采样，保护小数据集
    
    Args:
        features: 特征数组（可以是numpy数组或列表）
        labels: 原始事件标签列表
        max_samples_per_event: 每个事件的最大样本数（超过此数量的会降采样）
        min_samples_per_event: 每个事件的最小样本数（低于此数量的会保留全部）
        random_seed: 随机种子
    
    Returns:
        tuple: (采样后的特征, 采样后的标签, 采样索引映射)
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # 转换为numpy数组
    if isinstance(features, list):
        features = np.array(features)
    elif not isinstance(features, np.ndarray):
        features = np.array(features)
    
    # 按事件分组
    event_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        event_indices[label].append(idx)
    
    # 确定采样索引
    sampled_indices = []
    index_mapping = {}  # 原始索引 -> 新索引的映射
    
    # print(f"\n=== 基于先验分布的采样 ===")
    # print(f"最大样本数/事件: {max_samples_per_event}")
    # print(f"最小样本数/事件: {min_samples_per_event}")
    
    total_before = len(labels)
    total_after = 0
    
    for event, indices in event_indices.items():
        original_count = len(indices)
        
        if original_count > max_samples_per_event:
            # 大数据集：随机降采样
            sampled = np.random.choice(indices, size=max_samples_per_event, replace=False).tolist()
            # print(f"  事件 '{event}': {original_count} -> {max_samples_per_event} (降采样)")
        elif original_count < min_samples_per_event:
            # 小数据集：保留全部
            sampled = indices
            # print(f"  事件 '{event}': {original_count} -> {original_count} (保留全部)")
        else:
            # 中等数据集：保留全部
            sampled = indices
            # print(f"  事件 '{event}': {original_count} -> {original_count} (保留全部)")
        
        # 记录索引映射
        for orig_idx in sampled:
            new_idx = len(sampled_indices)
            index_mapping[orig_idx] = new_idx
            sampled_indices.append(orig_idx)
        
        total_after += len(sampled)
    
    # 按原始索引排序，保持顺序
    sampled_indices = sorted(sampled_indices)
    
    # 提取采样后的特征和标签
    sampled_features = features[sampled_indices]
    sampled_labels = [labels[i] for i in sampled_indices]
    
    # print(f"\n采样结果: {total_before} -> {total_after} 条样本 (保留 {total_after/total_before*100:.2f}%)")
    
    return sampled_features, sampled_labels, sampled_indices

def compute_sample_weights(labels, alpha=0.5):
    """
    基于先验分布计算样本权重，用于加权聚类
    
    Args:
        labels: 原始事件标签列表
        alpha: 权重调整参数 (0-1)，越大表示对少数类给予更多权重
    
    Returns:
        numpy array: 每个样本的权重
    """
    event_counts = Counter(labels)
    total_samples = len(labels)
    
    # 计算每个事件的权重（逆频率）
    event_weights = {}
    for event, count in event_counts.items():
        # 使用逆频率，alpha控制权重强度
        freq = count / total_samples
        weight = (1.0 / freq) ** alpha
        event_weights[event] = weight
    
    # 归一化权重
    max_weight = max(event_weights.values())
    for event in event_weights:
        event_weights[event] = event_weights[event] / max_weight
    
    # 为每个样本分配权重
    sample_weights = np.array([event_weights[label] for label in labels])
    
    # print(f"\n=== 样本权重计算 ===")
    # print(f"权重调整参数 alpha: {alpha}")
    # print(f"权重范围: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
    # print(f"平均权重: {sample_weights.mean():.4f}")
    
    return sample_weights
