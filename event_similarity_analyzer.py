#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件相似性分析程序
用于比较两个目录中的事件CSV文件，通过TF-IDF分析微博正文内容来判断事件相似性
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import jieba.posseg as pseg
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class EventSimilarityAnalyzer:
    def __init__(self, dir1_path, dir2_path):
        """
        初始化事件相似性分析器
        
        Args:
            dir1_path: 第一个结果文件目录路径
            dir2_path: 第二个结果文件目录路径
        """
        self.dir1_path = dir1_path
        self.dir2_path = dir2_path
        self.dir1_files = []
        self.dir2_files = []
        self.similarity_matrix = None
        self.matched_pairs = []
        
        # 存储四个维度的相似度矩阵
        self.when_similarity = None  # 时间维度
        self.where_similarity = None  # 地点维度
        self.what_similarity = None  # 事件维度
        self.who_similarity = None  # 人物维度
        
    def load_csv_files(self):
        """加载两个目录中的CSV文件"""
        print("正在加载CSV文件...")
        
        # 获取目录1中的CSV文件
        for file in os.listdir(self.dir1_path):
            if file.endswith('.csv'):
                self.dir1_files.append(file)
        
        # 获取目录2中的CSV文件
        for file in os.listdir(self.dir2_path):
            if file.endswith('.csv'):
                self.dir2_files.append(file)
        
        print(f"目录1中找到 {len(self.dir1_files)} 个CSV文件")
        print(f"目录2中找到 {len(self.dir2_files)} 个CSV文件")
        
    def preprocess_text(self, text):
        """
        文本预处理函数
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本
        """
        if pd.isna(text) or text == '':
            return ''
        
        # 转换为字符串
        text = str(text)
        
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 移除特殊字符，保留中文、英文、数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
        
        # 分词
        words = jieba.lcut(text)
        
        # 过滤停用词和短词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        words = [word for word in words if len(word) > 1 and word not in stop_words]
        
        return ' '.join(words)
    
    def extract_dimension_features(self, text, dimension='all'):
        """
        提取特定维度的特征
        
        Args:
            text: 原始文本
            dimension: 维度类型 ('when', 'where', 'what', 'who', 'all')
            
        Returns:
            提取的特征文本
        """
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 使用jieba进行词性标注
        words_with_pos = pseg.cut(text)
        
        extracted_words = []
        
        for word, pos in words_with_pos:
            # 过滤特殊字符
            if not re.match(r'[\u4e00-\u9fa5a-zA-Z0-9]+', word):
                continue
            
            if len(word) < 2:
                continue
            
            # 根据不同维度提取不同类型的词
            if dimension == 'when':
                # 时间维度：时间词
                if pos in ['TIME','t', 'm']:  # t:时间词, m:数词（可能表示日期）
                    extracted_words.append(word)
            
            elif dimension == 'where':
                # 地点维度：地点词
                if pos in ['ns', 'LOC']:  # ns:地名
                    extracted_words.append(word)
            
            elif dimension == 'what':
                # 事件维度：动词和名词
                if pos in ['v', 'vn', 'n', 'nz']:  # v:动词, vn:动名词, n:名词, nz:专有名词
                    extracted_words.append(word)
            
            elif dimension == 'who':
                # 人物维度：人名、机构名
                if pos in ['nr', 'nrfg', 'nrt', 'PER', 'ORG']:  # nr:人名, nrfg:国家名, nrt:职位名
                    extracted_words.append(word)
            
            elif dimension == 'all':
                # 全部维度
                if pos not in ['w', 'x', 'u', 'c', 'p', 'e', 'o']:  # 排除标点、助词等
                    extracted_words.append(word)
        
        # 过滤停用词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        extracted_words = [word for word in extracted_words if word not in stop_words]
        
        return ' '.join(extracted_words)
    
    def extract_tfidf_features(self, file_path, dimension='all'):
        """
        从CSV文件中提取TF-IDF特征
        
        Args:
            file_path: CSV文件路径
            dimension: 特征维度 ('when', 'where', 'what', 'who', 'all')
            
        Returns:
            特征文本
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path, encoding='utf-8')
            
            # 检查是否有微博正文字段
            if '微博正文' not in df.columns:
                print(f"警告: {file_path} 中没有找到'微博正文'字段")
                return None
            
            # 提取微博正文
            texts = df['微博正文'].dropna().tolist()
            
            if len(texts) == 0:
                print(f"警告: {file_path} 中没有有效的微博正文")
                return None
            
            # 根据维度提取特征
            if dimension == 'all':
                # 使用原来的预处理方法
                processed_texts = [self.preprocess_text(text) for text in texts]
            else:
                # 使用维度特征提取方法
                processed_texts = [self.extract_dimension_features(text, dimension) for text in texts]
            
            processed_texts = [text for text in processed_texts if text.strip()]
            
            if len(processed_texts) == 0:
                print(f"警告: {file_path} 在 {dimension} 维度预处理后没有有效文本")
                return None
            
            # 合并所有文本
            combined_text = ' '.join(processed_texts)
            
            return combined_text
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            return None
    
    def extract_multidimensional_features(self, file_path):
        """
        从CSV文件中提取多维度特征
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            包含四个维度特征的字典
        """
        return {
            'when': self.extract_tfidf_features(file_path, 'when'),
            'where': self.extract_tfidf_features(file_path, 'where'),
            'what': self.extract_tfidf_features(file_path, 'what'),
            'who': self.extract_tfidf_features(file_path, 'who')
        }
    
    def calculate_dimension_similarity(self, dir1_features, dir2_features, dimension_name):
        """
        计算特定维度的相似性矩阵
        
        Args:
            dir1_features: 目录1的特征列表
            dir2_features: 目录2的特征列表
            dimension_name: 维度名称
            
        Returns:
            相似性矩阵
        """
        if len(dir1_features) == 0 or len(dir2_features) == 0:
            print(f"警告: {dimension_name} 维度没有有效特征")
            return None
        
        # 合并所有特征用于TF-IDF计算
        all_features = dir1_features + dir2_features
        
        # 检查是否有足够的非空特征
        non_empty_features = [f for f in all_features if f and f.strip()]
        if len(non_empty_features) < 2:
            print(f"警告: {dimension_name} 维度的有效特征太少，跳过")
            return None
        
        # 计算TF-IDF
        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,  # 每个维度限制特征数量
                ngram_range=(1, 2),  # 使用1-gram和2-gram
                min_df=1,  # 最小文档频率
                max_df=0.9  # 最大文档频率
            )
            
            tfidf_matrix = vectorizer.fit_transform(all_features)
            
            # 分离两个目录的TF-IDF矩阵
            dir1_tfidf = tfidf_matrix[:len(dir1_features)]
            dir2_tfidf = tfidf_matrix[len(dir1_features):]
            
            # 计算相似性矩阵
            similarity_matrix = cosine_similarity(dir1_tfidf, dir2_tfidf)
            
            print(f"  {dimension_name} 维度相似性矩阵形状: {similarity_matrix.shape}")
            
            return similarity_matrix
            
        except Exception as e:
            print(f"  {dimension_name} 维度计算TF-IDF时出错: {str(e)}")
            return None
    
    def calculate_similarity_matrix(self):
        """计算两个目录中事件在四个维度的相似性矩阵"""
        print("正在计算多维度TF-IDF特征和相似性矩阵...")
        
        # 提取目录1的多维度特征
        dir1_features_when = []
        dir1_features_where = []
        dir1_features_what = []
        dir1_features_who = []
        dir1_file_names = []
        
        for file in self.dir1_files:
            file_path = os.path.join(self.dir1_path, file)
            features = self.extract_multidimensional_features(file_path)
            
            if any(features.values()):  # 至少有一个维度有特征
                dir1_features_when.append(features['when'] or '')
                dir1_features_where.append(features['where'] or '')
                dir1_features_what.append(features['what'] or '')
                dir1_features_who.append(features['who'] or '')
                dir1_file_names.append(file)
        
        # 提取目录2的多维度特征
        dir2_features_when = []
        dir2_features_where = []
        dir2_features_what = []
        dir2_features_who = []
        dir2_file_names = []
        
        for file in self.dir2_files:
            file_path = os.path.join(self.dir2_path, file)
            features = self.extract_multidimensional_features(file_path)
            
            if any(features.values()):  # 至少有一个维度有特征
                dir2_features_when.append(features['when'] or '')
                dir2_features_where.append(features['where'] or '')
                dir2_features_what.append(features['what'] or '')
                dir2_features_who.append(features['who'] or '')
                dir2_file_names.append(file)
        
        if len(dir1_file_names) == 0 or len(dir2_file_names) == 0:
            print("错误: 无法提取有效的文本特征")
            return
        
        print(f"成功提取 {len(dir1_file_names)} 个目录1文件和 {len(dir2_file_names)} 个目录2文件的特征")
        
        # 计算各个维度的相似性矩阵
        print("\n计算各维度相似性:")
        self.when_similarity = self.calculate_dimension_similarity(
            dir1_features_when, dir2_features_when, "时间(When)")
        
        self.where_similarity = self.calculate_dimension_similarity(
            dir1_features_where, dir2_features_where, "地点(Where)")
        
        self.what_similarity = self.calculate_dimension_similarity(
            dir1_features_what, dir2_features_what, "事件(What)")
        
        self.who_similarity = self.calculate_dimension_similarity(
            dir1_features_who, dir2_features_who, "人物(Who)")
        
        # 综合四个维度的相似性矩阵
        self.similarity_matrix = self.combine_similarity_matrices()
        
        self.dir1_file_names = dir1_file_names
        self.dir2_file_names = dir2_file_names
        
        if self.similarity_matrix is not None:
            print(f"\n综合相似性矩阵形状: {self.similarity_matrix.shape}")
    
    def combine_similarity_matrices(self, weights=None):
        """
        综合四个维度的相似性矩阵
        
        Args:
            weights: 各维度权重字典，如 {'when': 0.2, 'where': 0.2, 'what': 0.4, 'who': 0.2}
            
        Returns:
            综合相似性矩阵
        """
        if weights is None:
            # 默认权重：事件(what)权重最高
            weights = {'when': 0, 'where': 0, 'what': 1, 'who': 0}
        
        print(f"\n使用权重: 时间={weights['when']}, 地点={weights['where']}, 事件={weights['what']}, 人物={weights['who']}")
        
        matrices = []
        matrix_weights = []
        
        if self.when_similarity is not None:
            matrices.append(self.when_similarity)
            matrix_weights.append(weights['when'])
        
        if self.where_similarity is not None:
            matrices.append(self.where_similarity)
            matrix_weights.append(weights['where'])
        
        if self.what_similarity is not None:
            matrices.append(self.what_similarity)
            matrix_weights.append(weights['what'])
        
        if self.who_similarity is not None:
            matrices.append(self.who_similarity)
            matrix_weights.append(weights['who'])
        
        if len(matrices) == 0:
            print("错误: 没有有效的相似性矩阵")
            return None
        
        # 归一化权重
        total_weight = sum(matrix_weights)
        normalized_weights = [w / total_weight for w in matrix_weights]
        
        # 加权平均
        combined_matrix = np.zeros_like(matrices[0])
        for matrix, weight in zip(matrices, normalized_weights):
            combined_matrix += matrix * weight
        
        return combined_matrix
    
    def find_matched_pairs(self, threshold=0.3):
        """
        根据相似性矩阵找到匹配的事件对
        
        Args:
            threshold: 相似性阈值
        """
        if self.similarity_matrix is None:
            print("错误: 相似性矩阵未计算")
            return
        
        print(f"正在寻找相似性大于 {threshold} 的事件对...")
        
        matched_pairs = []
        used_dir2_indices = set()
        
        # 按相似性从高到低排序
        similarities = []
        for i in range(self.similarity_matrix.shape[0]):
            for j in range(self.similarity_matrix.shape[1]):
                similarities.append((i, j, self.similarity_matrix[i, j]))
        
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        # 选择最佳匹配
        for i, j, similarity in similarities:
            if similarity >= threshold and j not in used_dir2_indices:
                matched_pairs.append({
                    'dir1_file': self.dir1_file_names[i],
                    'dir2_file': self.dir2_file_names[j],
                    'similarity': similarity
                })
                used_dir2_indices.add(j)
        
        self.matched_pairs = matched_pairs
        print(f"找到 {len(matched_pairs)} 对匹配的事件")
        
        return matched_pairs
    
    def extract_event_name_from_filename(self, filename):
        """
        从文件名中提取事件名称（取第一个下划线之前的内容作为事件名）
        
        Args:
            filename: 文件名
            
        Returns:
            事件名称
        """
        # 移除.csv扩展名
        name = filename.replace('.csv', '')
        
        # 取第一个下划线之前的内容作为事件名
        event_name = name.split('_')[0]
        
        return event_name
    
    def get_dimension_similarity(self, i, j):
        """
        获取特定文件对在各维度的相似度
        
        Args:
            i: 目录1文件索引
            j: 目录2文件索引
            
        Returns:
            各维度相似度字典
        """
        similarities = {}
        
        if self.when_similarity is not None:
            similarities['when'] = self.when_similarity[i, j]
        else:
            similarities['when'] = 0.0
            
        if self.where_similarity is not None:
            similarities['where'] = self.where_similarity[i, j]
        else:
            similarities['where'] = 0.0
            
        if self.what_similarity is not None:
            similarities['what'] = self.what_similarity[i, j]
        else:
            similarities['what'] = 0.0
            
        if self.who_similarity is not None:
            similarities['who'] = self.who_similarity[i, j]
        else:
            similarities['who'] = 0.0
        
        return similarities
    
    def get_file_dimension_words(self, file_path, dimension):
        """
        获取文件在特定维度的分词结果
        
        Args:
            file_path: CSV文件路径
            dimension: 维度类型 ('when', 'where', 'what', 'who')
            
        Returns:
            分词列表
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path, encoding='utf-8')
            
            # 检查是否有微博正文字段
            if '微博正文' not in df.columns:
                return []
            
            # 提取微博正文
            texts = df['微博正文'].dropna().tolist()
            
            if len(texts) == 0:
                return []
            
            # 收集所有分词结果
            all_words = []
            
            for text in texts:
                if pd.isna(text) or text == '':
                    continue
                
                text = str(text)
                
                # 移除URL
                text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
                
                # 使用jieba进行词性标注
                words_with_pos = pseg.cut(text)
                
                for word, pos in words_with_pos:
                    # 过滤特殊字符
                    if not re.match(r'[\u4e00-\u9fa5a-zA-Z0-9]+', word):
                        continue
                    
                    if len(word) < 2:
                        continue
                    
                    # 根据不同维度提取不同类型的词
                    if dimension == 'when':
                        # 时间维度：时间词
                        if pos in ['TIME','t', 'm']:  # t:时间词, m:数词（可能表示日期）
                            all_words.append(word)
                    
                    elif dimension == 'where':
                        # 地点维度：地点词
                        if pos in ['ns', 'LOC']:  # ns:地名
                            all_words.append(word)
                    
                    elif dimension == 'what':
                        # 事件维度：动词和名词
                        if pos in ['v', 'vn', 'n', 'nz']:  # v:动词, vn:动名词, n:名词, nz:专有名词
                            all_words.append(word)
                    
                    elif dimension == 'who':
                        # 人物维度：人名、机构名
                        if pos in ['nr', 'nrfg', 'nrt', 'PER', 'ORG']:  # nr:人名, nrfg:国家名, nrt:职位名
                            all_words.append(word)
            
            # 过滤停用词
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
            all_words = [word for word in all_words if word not in stop_words]
            
            return all_words
            
        except Exception as e:
            print(f"提取文件 {file_path} 的 {dimension} 维度分词时出错: {str(e)}")
            return []
    
    def calculate_accuracy(self):
        """
        计算准确率
        
        Returns:
            准确率
        """
        if not self.matched_pairs:
            print("错误: 没有找到匹配的事件对")
            return 0.0
        
        correct_matches = 0
        total_matches = len(self.matched_pairs)
        
        print("\n匹配结果分析:")
        print("=" * 100)
        
        for i, pair in enumerate(self.matched_pairs, 1):
            dir1_event = self.extract_event_name_from_filename(pair['dir1_file'])
            dir2_event = self.extract_event_name_from_filename(pair['dir2_file'])
            similarity = pair['similarity']
            
            # 获取文件索引
            dir1_idx = self.dir1_file_names.index(pair['dir1_file'])
            dir2_idx = self.dir2_file_names.index(pair['dir2_file'])
            
            # 获取各维度相似度
            dim_sims = self.get_dimension_similarity(dir1_idx, dir2_idx)
            
            # 判断是否为同一事件（事件名称相同）
            is_correct = (dir1_event == dir2_event)
            if is_correct:
                correct_matches += 1
            
            status = "✓" if is_correct else "✗"
            print(f"{i:2d}. {status} 综合相似度: {similarity:.3f}")
            print(f"    When(时间)={dim_sims['when']:.3f}, Where(地点)={dim_sims['where']:.3f}, "
                  f"What(事件)={dim_sims['what']:.3f}, Who(人物)={dim_sims['who']:.3f}")
            print(f"    目录1: {pair['dir1_file']} -> 事件: {dir1_event}")
            print(f"    目录2: {pair['dir2_file']} -> 事件: {dir2_event}")
            
            # 打印所有维度的分词结果
            dimensions = ['when', 'where', 'what', 'who']
            dimension_names = {'when': '时间', 'where': '地点', 'what': '事件', 'who': '人物'}
            
            print("\n    [各维度分词结果]")
            for dim in dimensions:
                # 获取目录1文件的分词结果
                dir1_file_path = os.path.join(self.dir1_path, pair['dir1_file'])
                dir1_words = self.get_file_dimension_words(dir1_file_path, dim)
                
                # 获取目录2文件的分词结果
                dir2_file_path = os.path.join(self.dir2_path, pair['dir2_file'])
                dir2_words = self.get_file_dimension_words(dir2_file_path, dim)
                
                print(f"    {dimension_names[dim]}维度:")
                if dir1_words:
                    print(f"      目录1: {', '.join(set(dir1_words))}")
                else:
                    print(f"      目录1: [无]")
                
                if dir2_words:
                    print(f"      目录2: {', '.join(set(dir2_words))}")
                else:
                    print(f"      目录2: [无]")
            
            print()
        
        accuracy = correct_matches / total_matches if total_matches > 0 else 0.0
        
        print("=" * 100)
        print(f"总匹配对数: {total_matches}")
        print(f"正确匹配对数: {correct_matches}")
        print(f"准确率: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        return accuracy
    
    def save_results(self, output_file='event_similarity_results.txt'):
        """
        保存分析结果到文件
        
        Args:
            output_file: 输出文件名
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("事件相似性分析结果（多维度分析）\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"目录1路径: {self.dir1_path}\n")
            f.write(f"目录2路径: {self.dir2_path}\n")
            f.write(f"目录1文件数: {len(self.dir1_files)}\n")
            f.write(f"目录2文件数: {len(self.dir2_files)}\n\n")
            
            f.write("分析维度:\n")
            f.write("  - When(时间): 提取时间词、数词\n")
            f.write("  - Where(地点): 提取地名\n")
            f.write("  - What(事件): 提取动词、名词\n")
            f.write("  - Who(人物): 提取人名、机构名\n\n")
            
            if self.matched_pairs:
                f.write("匹配的事件对:\n")
                f.write("-" * 80 + "\n")
                
                correct_matches = 0
                for i, pair in enumerate(self.matched_pairs, 1):
                    dir1_event = self.extract_event_name_from_filename(pair['dir1_file'])
                    dir2_event = self.extract_event_name_from_filename(pair['dir2_file'])
                    is_correct = (dir1_event == dir2_event)
                    
                    # 获取各维度相似度
                    dir1_idx = self.dir1_file_names.index(pair['dir1_file'])
                    dir2_idx = self.dir2_file_names.index(pair['dir2_file'])
                    dim_sims = self.get_dimension_similarity(dir1_idx, dir2_idx)
                    
                    if is_correct:
                        correct_matches += 1
                    
                    status = "正确" if is_correct else "错误"
                    f.write(f"{i:2d}. [{status}] 综合相似度: {pair['similarity']:.3f}\n")
                    f.write(f"    维度相似度: When={dim_sims['when']:.3f}, Where={dim_sims['where']:.3f}, "
                           f"What={dim_sims['what']:.3f}, Who={dim_sims['who']:.3f}\n")
                    f.write(f"    目录1: {pair['dir1_file']} (事件: {dir1_event})\n")
                    f.write(f"    目录2: {pair['dir2_file']} (事件: {dir2_event})\n")
                    
                    # 保存所有维度的分词结果
                    dimensions = ['when', 'where', 'what', 'who']
                    dimension_names = {'when': '时间', 'where': '地点', 'what': '事件', 'who': '人物'}
                    
                    f.write("\n    [各维度分词结果]\n")
                    for dim in dimensions:
                        # 获取目录1文件的分词结果
                        dir1_file_path = os.path.join(self.dir1_path, pair['dir1_file'])
                        dir1_words = self.get_file_dimension_words(dir1_file_path, dim)
                        
                        # 获取目录2文件的分词结果
                        dir2_file_path = os.path.join(self.dir2_path, pair['dir2_file'])
                        dir2_words = self.get_file_dimension_words(dir2_file_path, dim)
                        
                        f.write(f"    {dimension_names[dim]}维度:\n")
                        if dir1_words:
                            f.write(f"      目录1: {', '.join(set(dir1_words))}\n")
                        else:
                            f.write(f"      目录1: [无]\n")
                        
                        if dir2_words:
                            f.write(f"      目录2: {', '.join(set(dir2_words))}\n")
                        else:
                            f.write(f"      目录2: [无]\n")
                    
                    f.write("\n")
                
                accuracy = correct_matches / len(self.matched_pairs)
                f.write("-" * 80 + "\n")
                f.write(f"准确率: {accuracy:.3f} ({accuracy*100:.1f}%)\n")
                f.write(f"总匹配对数: {len(self.matched_pairs)}\n")
                f.write(f"正确匹配对数: {correct_matches}\n")
        
        print(f"结果已保存到: {output_file}")
    
    def run_analysis(self, similarity_threshold=0.3):
        """
        运行完整的分析流程
        
        Args:
            similarity_threshold: 相似性阈值
        """
        print("开始事件相似性分析...")
        print("=" * 50)
        
        # 1. 加载CSV文件
        self.load_csv_files()
        
        # 2. 计算相似性矩阵
        self.calculate_similarity_matrix()
        
        if self.similarity_matrix is None:
            print("分析失败: 无法计算相似性矩阵")
            return
        
        # 3. 寻找匹配的事件对
        self.find_matched_pairs(similarity_threshold)
        
        # 4. 计算准确率
        accuracy = self.calculate_accuracy()
        
        # 5. 保存结果
        self.save_results()
        
        return accuracy


def main():
    """主函数"""
    # 设置目录路径
    dir1_path = "结果文件TEST1"
    dir2_path = "结果文件TEST2"
    
    # 创建分析器
    analyzer = EventSimilarityAnalyzer(dir1_path, dir2_path)
    
    # 运行分析
    accuracy = analyzer.run_analysis(similarity_threshold=0.3)
    
    print(f"\n分析完成！最终准确率: {accuracy:.3f}")


if __name__ == "__main__":
    main()
