#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微博事件关键词提取脚本
使用TF-IDF算法提取每个事件的关键词
"""

import os
import pandas as pd
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re
import json
from data_loader import load_weibo_data, preprocess_text
import numpy as np

class EventKeywordExtractor:
    def __init__(self, data_source_path=None, top_k=20):
        """
        初始化关键词提取器
        
        Args:
            data_source_path (str): 微博数据源路径
            top_k (int): 每个事件提取的关键词数量
        """
        self.data_source_path = data_source_path
        self.top_k = top_k
        self.stop_words = self._load_stop_words()
        self.events_data = {}
        
    def _load_stop_words(self):
        """加载中文停用词"""
        # 基础停用词列表
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '他', '她', '它', '们', '我们', '你们', '他们', '她们', '它们',
            '可以', '已经', '如果', '因为', '所以', '但是', '然后', '或者', '而且', '虽然', '不过', '只是', '就是', '还是', '就是', '还是', '就是', '还是', '就是', '还是', '就是', '还是', '就是', '还是', '就是', '还是', '就是', '还是',
            '什么', '怎么', '为什么', '哪里', '哪个', '谁', '多少', '几个', '一些', '很多', '非常', '特别', '比较', '更', '最', '太', '真', '真的', '确实', '当然', '肯定', '可能', '也许', '大概', '应该', '必须', '需要', '想要', '希望',
            '今天', '明天', '昨天', '现在', '以前', '以后', '时候', '时间', '地方', '这里', '那里', '这样', '那样', '这么', '那么', '这个', '那个', '这些', '那些', '这样', '那样', '这么', '那么', '这个', '那个', '这些', '那些',
            '微博', '转发', '评论', '点赞', '关注', '粉丝', '用户', '账号', '发布', '内容', '图片', '视频', '链接', '话题', '标签', '超话', '热搜', '热门', '推荐', '发现', '搜索', '分享', '收藏', '举报', '屏蔽', '删除', '编辑',
            'http', 'https', 'www', 'com', 'cn', 'org', 'net', 'html', 'php', 'asp', 'jsp', 'xml', 'json', 'css', 'js', 'jpg', 'png', 'gif', 'mp4', 'avi', 'mov', 'wmv', 'flv', 'mp3', 'wav', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx',
            'nan', 'null', 'none', 'undefined', 'true', 'false', 'yes', 'no', 'ok', 'okay', 'thanks', 'thank', 'please', 'sorry', 'hello', 'hi', 'bye', 'goodbye', 'see', 'you', 'later', 'soon', 'now', 'here', 'there', 'where', 'when', 'how', 'what', 'who', 'why', 'which', 'whose', 'whom'
        }
        
        # 尝试从文件加载更多停用词
        try:
            # 常见的停用词文件路径
            stop_word_files = [
                'stopwords.txt',
                'chinese_stopwords.txt',
                'stop_words.txt',
                'stopwords_zh.txt'
            ]
            
            for file_path in stop_word_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        additional_stop_words = set(line.strip() for line in f if line.strip())
                        stop_words.update(additional_stop_words)
                    print(f"从 {file_path} 加载了 {len(additional_stop_words)} 个停用词")
                    break
        except Exception as e:
            print(f"加载停用词文件时出错: {e}")
            
        return stop_words
    
    def _preprocess_text_for_keywords(self, text):
        """
        为关键词提取预处理文本
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 预处理后的文本
        """
        if not isinstance(text, str):
            return ""
            
        # 移除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 移除@用户名
        text = re.sub(r'@\w+', '', text)
        
        # 移除#话题#标签
        text = re.sub(r'#.*?#', '', text)
        
        # 移除多余的空格和换行
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _segment_text(self, text):
        """
        中文分词，只保留名词、动词、形容词等有意义的词性
        
        Args:
            text (str): 输入文本
            
        Returns:
            list: 分词结果
        """
        if not text.strip():
            return []
            
        # 使用jieba进行词性标注
        words = pseg.cut(text)
        
        # 只保留有意义的词性
        meaningful_pos = {'n', 'v', 'a', 'nr', 'ns', 'nt', 'nw', 'nz', 'vn', 'an', 'ad', 'ag', 'al', 'b', 'f', 'm', 'q', 'r', 's', 't', 'x', 'y', 'z'}
        
        filtered_words = []
        for word, pos in words:
            # 过滤条件：
            # 1. 词长度大于1
            # 2. 不在停用词中
            # 3. 词性在允许的范围内
            # 4. 不是纯数字或纯符号
            if (len(word) > 1 and 
                word not in self.stop_words and 
                pos in meaningful_pos and
                not re.match(r'^[\d\s#@,.!?，。！？、]+$', word) and
                re.search(r'[\u4e00-\u9fff]', word)):  # 至少包含一个中文字符
                filtered_words.append(word)
                
        return filtered_words
    
    def load_events_data(self):
        """加载所有事件的数据"""
        print("正在加载微博数据...")
        
        if self.data_source_path is None:
            # 使用默认路径
            self.data_source_path = "./cluster_events"
            
        if not os.path.exists(self.data_source_path):
            raise ValueError(f"数据源路径不存在: {self.data_source_path}")
            
        # 使用data_loader加载数据
        texts, image_paths_groups, events, all_data_df = load_weibo_data(
            data_source_path=self.data_source_path
        )
        
        # 按事件分组数据
        for i, (text, event) in enumerate(zip(texts, events)):
            if event not in self.events_data:
                self.events_data[event] = []
            
            # 预处理文本用于关键词提取
            processed_text = self._preprocess_text_for_keywords(text)
            if processed_text.strip():
                self.events_data[event].append(processed_text)
        
        print(f"成功加载 {len(self.events_data)} 个事件的数据")
        for event, texts in self.events_data.items():
            print(f"  - {event}: {len(texts)} 条文本")
    
    def extract_keywords_tfidf(self, event_name, texts):
        """
        使用TF-IDF提取单个事件的关键词
        
        Args:
            event_name (str): 事件名称
            texts (list): 该事件的文本列表
            
        Returns:
            list: 关键词列表
        """
        if not texts:
            return []
            
        # 对每个文本进行分词
        segmented_texts = []
        for text in texts:
            words = self._segment_text(text)
            if words:
                segmented_texts.append(' '.join(words))
        
        if not segmented_texts:
            return []
            
        try:
            # 使用TF-IDF向量化
            vectorizer = TfidfVectorizer(
                max_features=1000,  # 最大特征数
                min_df=2,           # 最小文档频率
                max_df=0.8,         # 最大文档频率
                ngram_range=(1, 2), # 1-gram和2-gram
                token_pattern=r'\S+' # 匹配非空白字符
            )
            
            tfidf_matrix = vectorizer.fit_transform(segmented_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # 计算每个词的TF-IDF分数
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # 获取top-k关键词
            top_indices = np.argsort(mean_scores)[-self.top_k:][::-1]
            keywords = [(feature_names[i], mean_scores[i]) for i in top_indices if mean_scores[i] > 0]
            
            return keywords
            
        except Exception as e:
            print(f"提取事件 {event_name} 关键词时出错: {e}")
            return []
    
    def extract_keywords_frequency(self, event_name, texts):
        """
        使用词频统计提取关键词（备用方法）
        
        Args:
            event_name (str): 事件名称
            texts (list): 该事件的文本列表
            
        Returns:
            list: 关键词列表
        """
        if not texts:
            return []
            
        # 统计所有词的频率
        word_freq = Counter()
        for text in texts:
            words = self._segment_text(text)
            word_freq.update(words)
        
        # 返回频率最高的词
        return word_freq.most_common(self.top_k)
    
    def extract_all_keywords(self, method='tfidf'):
        """
        提取所有事件的关键词
        
        Args:
            method (str): 提取方法，'tfidf' 或 'frequency'
            
        Returns:
            dict: 每个事件的关键词字典
        """
        if not self.events_data:
            self.load_events_data()
            
        all_keywords = {}
        
        print(f"\n使用 {method} 方法提取关键词...")
        
        for event_name, texts in self.events_data.items():
            print(f"正在处理事件: {event_name}")
            
            if method == 'tfidf':
                keywords = self.extract_keywords_tfidf(event_name, texts)
            else:
                keywords = self.extract_keywords_frequency(event_name, texts)
                
            all_keywords[event_name] = keywords
            print(f"  提取到 {len(keywords)} 个关键词")
            
        return all_keywords
    
    def save_keywords(self, keywords_dict, output_file='event_keywords.json'):
        """
        保存关键词到文件
        
        Args:
            keywords_dict (dict): 关键词字典
            output_file (str): 输出文件名
        """
        # 转换为可序列化的格式
        serializable_dict = {}
        for event, keywords in keywords_dict.items():
            if isinstance(keywords[0], tuple):
                # TF-IDF结果 (word, score)
                serializable_dict[event] = [
                    {"word": word, "score": float(score)} 
                    for word, score in keywords
                ]
            else:
                # 频率统计结果 (word, count)
                serializable_dict[event] = [
                    {"word": word, "count": int(count)} 
                    for word, count in keywords
                ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_dict, f, ensure_ascii=False, indent=2)
        
        print(f"关键词已保存到: {output_file}")
    
    def save_simple_keywords(self, keywords_dict, output_file='simple_top_10_keywords.json', top_n=10):
        """
        保存简化版关键词到文件（只包含关键词文本，不包含分数）
        
        Args:
            keywords_dict (dict): 关键词字典
            output_file (str): 输出文件名
            top_n (int): 每个事件提取的关键词数量
        """
        # 创建简化的数据结构，只包含关键词文本
        simple_keywords = {}
        
        for event_name, keywords in keywords_dict.items():
            if isinstance(keywords[0], tuple):
                # TF-IDF结果 (word, score)
                simple_keywords[event_name] = [word for word, score in keywords[:top_n]]
            else:
                # 频率统计结果 (word, count)
                simple_keywords[event_name] = [word for word, count in keywords[:top_n]]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(simple_keywords, f, ensure_ascii=False, indent=2)
        
        print(f"简化版关键词已保存到: {output_file}")
        return simple_keywords

    def print_keywords_summary(self, keywords_dict):
        """
        打印关键词摘要
        
        Args:
            keywords_dict (dict): 关键词字典
        """
        print("\n" + "="*80)
        print("事件关键词提取结果摘要")
        print("="*80)
        
        for event_name, keywords in keywords_dict.items():
            print(f"\n事件: {event_name}")
            print("-" * 50)
            
            if not keywords:
                print("  未提取到关键词")
                continue
                
            if isinstance(keywords[0], tuple):
                # TF-IDF结果
                for i, (word, score) in enumerate(keywords[:10], 1):  # 只显示前10个
                    print(f"  {i:2d}. {word:<15} (TF-IDF: {score:.4f})")
            else:
                # 频率统计结果
                for i, (word, count) in enumerate(keywords[:10], 1):  # 只显示前10个
                    print(f"  {i:2d}. {word:<15} (频次: {count})")
            
            if len(keywords) > 10:
                print(f"  ... 还有 {len(keywords) - 10} 个关键词")

def main():
    """主函数"""
    print("微博事件关键词提取工具")
    print("="*50)
    
    # 配置参数
    data_source_path = "./结果文件_61"  # 可以修改为其他路径
    top_k = 20  # 每个事件提取的关键词数量
    
    # 创建提取器
    extractor = EventKeywordExtractor(
        data_source_path=data_source_path,
        top_k=top_k
    )
    
    try:
        # 提取关键词（使用TF-IDF方法）
        print("使用TF-IDF方法提取关键词...")
        tfidf_keywords = extractor.extract_all_keywords(method='tfidf')
        
        # 保存结果
        extractor.save_keywords(tfidf_keywords, 'event_keywords_tfidf.json')
        
        # 保存简化版关键词（只包含关键词文本）
        simple_keywords = extractor.save_simple_keywords(tfidf_keywords, 'simple_top_10_keywords.json', top_n=10)
        
        # 打印摘要
        extractor.print_keywords_summary(tfidf_keywords)
        
        # 可选：也使用频率统计方法
        print("\n" + "="*80)
        print("使用词频统计方法提取关键词...")
        freq_keywords = extractor.extract_all_keywords(method='frequency')
        
        # 保存频率统计结果
        extractor.save_keywords(freq_keywords, 'event_keywords_frequency.json')
        
        print("\n关键词提取完成！")
        print("结果文件:")
        print("  - event_keywords_tfidf.json (TF-IDF方法，完整结果)")
        print("  - event_keywords_frequency.json (词频统计方法)")
        print("  - simple_top_10_keywords.json (每个事件最重要的10个关键词)")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()