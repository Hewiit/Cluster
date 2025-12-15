import os
import pandas as pd
from PIL import Image
import re
from config import *

# Twitter100k数据加载函数（已注释）
# def load_data():
#     """加载Twitter100k文本和图像数据"""
#     # 加载文本数据
#     with open(TEXT_FILE_PATH, 'r', encoding='utf-8') as f:
#         texts = [line.strip() for line in f]
#
#     # 加载图像数据路径
#     image_paths = sorted(
#         [os.path.join(IMAGE_DIR, fname) for fname in os.listdir(IMAGE_DIR) 
#          if fname.endswith(('.jpg', '.png', '.jpeg'))])
#     
#     assert len(texts) == len(image_paths), "文本和图像的数量不匹配，请检查数据集。"
#     return texts, image_paths

def preprocess_text(text):
    """预处理文本：清理特殊字符和表情符号，限制长度"""
    if not isinstance(text, str):
        return "空文本"
    
    # 移除URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 移除@用户名
    text = re.sub(r'@\w+', '', text)
    
    # 保留中文字符、英文字母、数字和基本标点
    # 注意：不再过滤掉非ASCII字符，以保留中文
    filtered_text = re.sub(r'[^\u4e00-\u9fff\w\s,.!?，。！？、]', '', text)
    
    # 检查是否只包含数字和特殊字符
    if re.match(r'^[\d\s#@,.!?，。！？、]+$', filtered_text.strip()):
        # 如果只有数字和特殊字符，检查原始文本是否有更多信息
        if len(text) > len(filtered_text) and re.search(r'[\u4e00-\u9fff]', text):
            # 如果原始文本包含中文，使用更宽松的过滤
            filtered_text = re.sub(r'[^\u4e00-\u9fff\w\s,.!?，。！？、]', '', text)
        else:
            # 添加标记以便于识别
            filtered_text = "数字内容: " + filtered_text.strip()
    
    # 确保文本不为空
    if not filtered_text.strip():
        return "空文本"
    
    # 移除多余的空格
    filtered_text = ' '.join(filtered_text.split())
    
    # 限制文本长度（按词数）
    words = filtered_text.split()
    if len(words) > 100:  # 设置合理的长度限制
        filtered_text = ' '.join(words[:100])
    
    return filtered_text.strip()

def load_crisis_data():
    """加载Crisis数据集的文本和图像数据，每个事件限制500条"""
    texts = []
    image_paths = []
    labels = []
    
    # 遍历所有灾难事件
    for event in CRISIS_EVENTS:
        # 读取TSV文件
        tsv_path = os.path.join(CRISIS_DATA_DIR, f"{event}_final_data.tsv")
        df = pd.read_csv(tsv_path, sep='\t', nrows=500)  # 只读取前500行
        # df = pd.read_csv(tsv_path, sep='\t')  # 读取所有行
        
        # 获取文本和图像路径，并预处理文本
        event_texts = [preprocess_text(text) for text in df['tweet_text'].tolist()]
        event_image_paths = [os.path.join(CRISIS_IMAGE_DIR, path) for path in df['image_path'].tolist()]
        
        # 过滤掉空文本
        valid_indices = [i for i, text in enumerate(event_texts) if text.strip()]
        event_texts = [event_texts[i] for i in valid_indices]
        event_image_paths = [event_image_paths[i] for i in valid_indices]
        
        print(f"事件 {event} 加载了 {len(event_texts)} 条有效数据")
        
        texts.extend(event_texts)
        image_paths.extend(event_image_paths)
        labels.extend([event] * len(event_texts))
    
    print(f"\n总共加载了 {len(texts)} 条数据")
    assert len(texts) == len(image_paths) == len(labels), "文本、图像和标签的数量不匹配"
    return texts, image_paths, labels

def load_image(image_path, preprocess):
    """加载并预处理单个图像"""
    try:
        image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(DEVICE)
        return image
    except Exception as e:
        print(f"无法加载图像 {image_path}: {str(e)}")
        return None 

def _extract_source_site_from_filename(filename):
    """根据CSV文件名提取来源网站（假设以_{site}.csv结尾）"""
    name = os.path.splitext(os.path.basename(filename))[0]
    if '_' in name:
        return name.split('_')[-1]
    return ""

def load_all_platform_data(source_site=None, data_source_path=None):
    """加载ALL_PLATFORM数据集，只包含文本模态
    
    Args:
        source_site (str, optional): 指定来源网站（如“懂车帝”）。若为None则加载所有来源
        data_source_path (str, optional): 数据源路径。如果为None，使用config中的ALL_PLATFORM_BASE_DIR
    """
    texts = []
    image_paths_groups = []
    labels = []
    
    base_dir = data_source_path if data_source_path is not None else ALL_PLATFORM_BASE_DIR
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"ALL_PLATFORM数据目录不存在: {base_dir}")
    
    event_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not event_dirs:
        raise ValueError(f"在目录 {base_dir} 中未找到任何事件子目录")
    
    for event in event_dirs:
        event_path = os.path.join(base_dir, event)
        csv_files = [f for f in os.listdir(event_path) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"提示：事件 {event} 中未找到CSV文件，跳过")
            continue
        
        # 每个事件目录下只有一个CSV文件，直接使用第一个
        csv_file = csv_files[0]
        
        # 如果指定了来源网站，检查来源网站名是否在CSV文件名中存在
        if source_site:
            csv_file_lower = csv_file.lower()
            source_site_lower = source_site.lower()
            if source_site_lower not in csv_file_lower:
                print(f"提示：事件 {event} 的CSV文件名中不包含来源网站 '{source_site}'，跳过")
                continue
        
        csv_path = os.path.join(event_path, csv_file)
        event_label = event  # 使用事件目录名作为标签
        print(f"正在处理事件 {event}（CSV文件: {csv_file}）")
        
        try:
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding='gb18030')
        except Exception as e:
            print(f"读取CSV文件失败: {csv_path}, 错误: {str(e)}")
            continue
        
        missing_fields = [field for field in ALL_PLATFORM_EXPECTED_FIELDS if field not in df.columns]
        if missing_fields:
            print(f"警告：文件 {csv_file} 缺少字段 {missing_fields}，将尽可能处理存在的列")
        
        # 查找两个文本字段
        text_column1 = None
        text_column2 = None
        for col in ALL_PLATFORM_TEXT_FIELDS:
            if col in df.columns:
                if text_column1 is None:
                    text_column1 = col
                elif text_column2 is None:
                    text_column2 = col
                    break
        
        if text_column1 is None:
            print(f"警告：文件 {csv_file} 中未找到文本字段，跳过")
            continue
        
        valid_count = 0
        for _, row in df.iterrows():
            # 组合两个文本字段的内容
            text_parts = []
            if text_column1:
                text1 = str(row[text_column1]) if text_column1 in row else ""
                if text1 and text1.strip() and text1 != "nan":
                    text_parts.append(text1.strip())
            if text_column2:
                text2 = str(row[text_column2]) if text_column2 in row else ""
                if text2 and text2.strip() and text2 != "nan":
                    # 避免重复内容
                    if text2.strip() not in text_parts:
                        text_parts.append(text2.strip())
            
            # 组合文本字段（用换行符分隔，或直接拼接）
            raw_text = "\n".join(text_parts) if text_parts else ""
            text = preprocess_text(raw_text)
            if not text.strip():
                continue
            
            texts.append(text)
            image_paths_groups.append([])  # ALL_PLATFORM数据集无图像模态
            labels.append(event_label)
            valid_count += 1
        
        print(f"事件 {event} 成功处理 {valid_count} 条文本数据")
    
    if not texts:
        raise ValueError("未能从ALL_PLATFORM数据集中加载任何有效文本")
    
    print(f"\nALL_PLATFORM数据加载完成，共 {len(texts)} 条文本，覆盖 {len(set(labels))} 个文件标签")
    if source_site:
        print(f"来源网站过滤条件：{source_site}")
    
    return texts, image_paths_groups, labels

def load_weibo_data(date=None, data_source_path=None):
    """加载微博数据集，包括只有文本的微博和真实标签
    
    Args:
        date (str, optional): 指定日期，格式为YYYY-MM-DD。如果提供，将只读取{event}_{date}.csv文件
        data_source_path (str, optional): 数据源路径。如果为None，使用config中的WEIBO_BASE_DIR
    """
    all_texts = []
    all_image_paths_groups = []
    all_events = []  # 用作真实标签
    processed_events = []  # 记录成功处理的事件
    
    # 使用指定的数据源路径或默认路径
    base_dir = data_source_path if data_source_path is not None else WEIBO_BASE_DIR
    
    # 获取事件列表
    event_dirs = [d for d in os.listdir(base_dir) 
                 if os.path.isdir(os.path.join(base_dir, d))]
    
    # 计算每个事件的数据量
    for event in event_dirs:
        event_path = os.path.join(base_dir, event)
        image_dir = os.path.join(event_path, WEIBO_IMAGE_DIRNAME)
        
        # 即使没有图片目录也继续处理
        has_image_dir = os.path.exists(image_dir)
        if not has_image_dir:
            print(f"提示：事件 {event} 中未找到图片目录，将只处理文本")
        
        # 根据是否指定日期来选择CSV文件
        if date:
            # 如果指定了日期，查找{event}_{date}.csv文件
            csv_file = f"{event}_{date}.csv"
            csv_path = os.path.join(event_path, csv_file)
            
            # 检查文件是否存在
            if not os.path.exists(csv_path):
                print(f"跳过事件 {event}：未找到日期 {date} 的数据文件 {csv_file}")
                continue
        else:
            # 如果没有指定日期，使用原来的逻辑
            csv_files = [f for f in os.listdir(event_path) 
                        if f.endswith('.csv')]
            
            if not csv_files:
                print(f"警告：在事件 {event} 中未找到CSV文件")
                continue
                
            # 优先使用与事件名称匹配的CSV文件，否则使用第一个CSV文件
            csv_file = next((f for f in csv_files if f.lower().startswith(event.lower())), 
                           csv_files[0])
            csv_path = os.path.join(event_path, csv_file)
        
        # 使用CSV文件名（不含扩展名）作为事件标签
        event_label = os.path.splitext(csv_file)[0]
            
        try:
            print(f"正在处理事件 {event_label} 的数据文件: {csv_file}")
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            if not all(field in df.columns for field in WEIBO_REQUIRED_FIELDS.values()):
                print(f"警告：事件 {event_label} 的CSV文件缺少必要字段")
                print(f"需要的字段: {list(WEIBO_REQUIRED_FIELDS.values())}")
                print(f"实际的字段: {list(df.columns)}")
                continue
                
            text_only_count = 0
            multimodal_count = 0
            
            for _, row in df.iterrows():
                text = str(row[WEIBO_REQUIRED_FIELDS['text']])
                text = preprocess_text(text)
                if not text.strip():
                    continue
                    
                base_img_id = str(row[WEIBO_REQUIRED_FIELDS['id']]).strip()
                image_paths_group = []
                
                # 如果有图片目录，尝试收集图片
                if has_image_dir and base_img_id:
                    found_first = False
                    found_zero = False
                    for ext in ['.jpg', '.jpeg', '.png']:
                        img_path = os.path.join(image_dir, f"{base_img_id}{ext}")
                        img_path2 = os.path.join(image_dir, f"{base_img_id}-0{ext}")
                        # 只添加存在的图片且不重复
                        if os.path.exists(img_path):
                            image_paths_group.append(img_path)
                            found_first = True
                        if os.path.exists(img_path2) and img_path2 != img_path:
                            image_paths_group.append(img_path2)
                            found_first = True
                            found_zero = True
                        if found_first:
                            break

                    # 只有找到-0结尾的图片时才检查后续图片（ID-1, ID-2, ...）
                    if found_zero:
                        for i in range(1, 10):
                            for ext in ['.jpg', '.jpeg', '.png']:
                                img_path = os.path.join(image_dir, f"{base_img_id}-{i}{ext}")
                                if os.path.exists(img_path):
                                    image_paths_group.append(img_path)
                                    break
                                else:
                                    break
                
                # 添加数据，无论是否有图片
                all_texts.append(text)
                all_image_paths_groups.append(image_paths_group)
                all_events.append(event_label)  # 使用事件标签
                
                if image_paths_group:
                    multimodal_count += 1
                else:
                    text_only_count += 1
            
            # 记录成功处理的事件
            processed_events.append(event_label)
            print(f"事件 {event_label} 成功处理 {multimodal_count} 条多模态数据，{text_only_count} 条纯文本数据")
                        
        except Exception as e:
            print(f"处理事件 {event_label} 时出错: {str(e)}")
            continue
    
    if not all_texts:
        raise ValueError("未能成功加载任何数据")
        
    total_samples = len(all_texts)
    total_events = len(set(all_events))
    processed_event_count = len(processed_events)
    multimodal_samples = sum(1 for paths in all_image_paths_groups if paths)
    text_only_samples = total_samples - multimodal_samples
    
    print("\n数据加载统计：")
    if date:
        print(f"日期 {date} 的数据统计：")
    print(f"总共加载 {total_samples} 条微博数据，来自 {total_events} 个事件类型")
    print(f"成功处理了 {processed_event_count} 个事件目录的数据")
    print(f"其中多模态数据 {multimodal_samples} 条，纯文本数据 {text_only_samples} 条")
    
    # 多图微博统计
    image_counts = [len(group) for group in all_image_paths_groups if group]
    if image_counts:
        multi_image_count = sum(1 for count in image_counts if count > 1)
        avg_images = sum(image_counts) / len(image_counts)
        print(f"在多模态数据中：")
        print(f"- 包含 {multi_image_count} 条多图微博")
        print(f"- 平均每条微博的图片数量: {avg_images:.2f}")
    
    return all_texts, all_image_paths_groups, all_events

def load_data(date=None, data_source_path=None, source_site=None):
    """根据配置选择加载不同的数据集
    
    Args:
        date (str, optional): 指定日期，格式为YYYY-MM-DD。仅对WEIBO数据集有效
        data_source_path (str, optional): 数据源路径。如果为None，使用config中的默认路径
        source_site (str, optional): 指定ALL_PLATFORM数据集的来源网站过滤条件
    """
    if DATASET_TYPE == "CRISIS":
        return load_crisis_data()
    elif DATASET_TYPE == "WEIBO":
        return load_weibo_data(date=date, data_source_path=data_source_path)
    elif DATASET_TYPE == "ALL_PLATFORM":
        return load_all_platform_data(source_site=source_site, data_source_path=data_source_path)
    elif DATASET_TYPE == "TWITTER100K":
        raise NotImplementedError("Twitter100k数据集加载功能已被注释")
    else:
        raise ValueError(f"未知的数据集类型: {DATASET_TYPE}") 