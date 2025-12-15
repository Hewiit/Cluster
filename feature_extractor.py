import numpy as np
import torch
import re
from PIL import Image
from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
    CLIPImageProcessor,
    CLIPVisionModel
)
import warnings
import os
from config import *
from data_loader import load_image

_FEATURE_EXTRACTOR_SINGLETON = None

def get_local_model_path(model_name, local_model_dir="./models"):
    """
    获取本地模型路径
    
    Args:
        model_name: 模型名称，如 "bert-base-chinese" 或 "hfl/chinese-bert-wwm-ext" 或 "openai/clip-vit-base-patch32"
        local_model_dir: 本地模型目录
    
    Returns:
        本地模型路径，如果不存在则返回None
    """
    # 处理包含组织名的模型（如 "hfl/chinese-bert-wwm-ext"）
    if '/' in model_name:
        org, model = model_name.split('/', 1)
        # 尝试两种路径格式：org/model 和 model
        local_path1 = os.path.join(local_model_dir, org, model)
        local_path2 = os.path.join(local_model_dir, model)
        
        if os.path.exists(local_path1):
            return local_path1
        elif os.path.exists(local_path2):
            return local_path2
    else:
        local_path = os.path.join(local_model_dir, model_name)
        if os.path.exists(local_path):
            return local_path
    
    return None

class FeatureExtractor:
    def __init__(self):
        # 忽略特定的警告
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 模型配置
        self.local_model_dir = "./models"
        
        # 确保模型目录存在
        if not os.path.exists(self.local_model_dir):
            print(f"警告：本地模型目录 {self.local_model_dir} 不存在，将尝试从网络加载")
        
        # 设置环境变量以指定backbone路径
        os.environ['TORCH_HOME'] = os.path.abspath(self.local_model_dir)
        
        # 初始化文本编码器
        self._init_text_encoder()
        
        # 初始化图像编码器
        self._init_image_encoder()
        
        # 初始化NER模型
        self._init_ner_model()
        
        # 添加模态转换网络
        self.modality_transform = torch.nn.Sequential(
            torch.nn.Linear(DESIRED_DIM, DESIRED_DIM),
            torch.nn.ReLU(),
            torch.nn.Linear(DESIRED_DIM, DESIRED_DIM),
            torch.nn.LayerNorm(DESIRED_DIM)
        ).to(DEVICE)
        
        # 添加NER特征转换层
        self.ner_transform = torch.nn.Linear(
            self.ner_model.config.hidden_size, 
            DESIRED_DIM
        ).to(DEVICE)
        
        # 添加动态特征转换层，用于处理不同维度的特征
        self.text_dim_transform = {}  # 缓存不同维度的转换层
        self.entity_dim_transform = {}  # 缓存不同维度的转换层
        self.image_dim_transform = {}  # 缓存不同维度的转换层
        
        # 设置所有模型为评估模式
        self.text_model.eval()
        self.image_model.eval()
        self.ner_model.eval()
        self.modality_transform.eval()
        self.ner_transform.eval()
        
        # 设置所有动态转换层为eval模式
        for transform_layer in self.text_dim_transform.values():
            transform_layer.eval()
        for transform_layer in self.entity_dim_transform.values():
            transform_layer.eval()
        for transform_layer in self.image_dim_transform.values():
            transform_layer.eval()

    def _init_text_encoder(self):
        """初始化文本编码器"""
        print("初始化文本编码器...")
        
        try:
            if DATASET_TYPE in ["CRISIS", "TWITTER100K"]:
                # 英文文本编码器
                model_name = "bert-base-uncased"
                local_path = get_local_model_path(model_name, self.local_model_dir)
                if local_path:
                    print(f"从本地路径加载模型: {local_path}")
                    self.text_model = AutoModel.from_pretrained(local_path).to(DEVICE)
                    self.text_tokenizer = AutoTokenizer.from_pretrained(local_path)
                else:
                    print(f"本地模型不存在，从网络加载: {model_name}")
                    self.text_model = AutoModel.from_pretrained(model_name).to(DEVICE)
                    self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)
                print("成功加载英文文本编码器")
            else:
                # 中文文本编码器
                model_name = "bert-base-chinese"
                local_path = get_local_model_path(model_name, self.local_model_dir)
                if local_path:
                    print(f"从本地路径加载模型: {local_path}")
                    self.text_model = AutoModel.from_pretrained(local_path).to(DEVICE)
                    self.text_tokenizer = AutoTokenizer.from_pretrained(local_path)
                else:
                    print(f"本地模型不存在，从网络加载: {model_name}")
                    self.text_model = AutoModel.from_pretrained(model_name).to(DEVICE)
                    self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)
                print("成功加载中文文本编码器")
                    
        except Exception as e:
            print(f"加载文本编码器失败: {str(e)}")
            raise

    def _init_image_encoder(self):
        """初始化图像编码器"""
        print("初始化图像编码器...")
        
        try:
            # 使用CLIP的图像编码器部分
            model_name = "openai/clip-vit-base-patch32"
            local_path = get_local_model_path(model_name, self.local_model_dir)
            if local_path:
                print(f"从本地路径加载模型: {local_path}")
                self.image_model = CLIPVisionModel.from_pretrained(local_path).to(DEVICE)
                self.image_processor = CLIPImageProcessor.from_pretrained(local_path)
            else:
                print(f"本地模型不存在，从网络加载: {model_name}")
                self.image_model = CLIPVisionModel.from_pretrained(model_name).to(DEVICE)
                self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
            print("成功加载CLIP图像编码器")
                
        except Exception as e:
            print(f"加载图像编码器失败: {str(e)}")
            raise

    def _init_ner_model(self):
        """初始化NER模型"""
        print("初始化NER模型...")
        
        try:
            if DATASET_TYPE in ["CRISIS", "TWITTER100K"]:
                # 英文NER模型
                model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
                local_path = get_local_model_path(model_name, self.local_model_dir)
                if local_path:
                    print(f"从本地路径加载模型: {local_path}")
                    self.ner_model = AutoModelForTokenClassification.from_pretrained(
                        local_path,
                        output_hidden_states=True,
                        ignore_mismatched_sizes=True
                    ).to(DEVICE)
                    self.ner_tokenizer = AutoTokenizer.from_pretrained(local_path)
                else:
                    print(f"本地模型不存在，从网络加载: {model_name}")
                    self.ner_model = AutoModelForTokenClassification.from_pretrained(
                        model_name,
                        output_hidden_states=True,
                        ignore_mismatched_sizes=True
                    ).to(DEVICE)
                    self.ner_tokenizer = AutoTokenizer.from_pretrained(model_name)
                print("成功加载英文NER模型")
            else:
                # 中文NER模型
                model_name = "hfl/chinese-bert-wwm-ext"
                local_path = get_local_model_path(model_name, self.local_model_dir)
                if local_path:
                    print(f"从本地路径加载模型: {local_path}")
                    self.ner_model = AutoModelForTokenClassification.from_pretrained(
                        local_path,
                        output_hidden_states=True,
                        ignore_mismatched_sizes=True
                    ).to(DEVICE)
                    self.ner_tokenizer = AutoTokenizer.from_pretrained(local_path)
                else:
                    print(f"本地模型不存在，从网络加载: {model_name}")
                    self.ner_model = AutoModelForTokenClassification.from_pretrained(
                        model_name,
                        output_hidden_states=True,
                        ignore_mismatched_sizes=True
                    ).to(DEVICE)
                    self.ner_tokenizer = AutoTokenizer.from_pretrained(model_name)
                print("成功加载中文NER模型")
            
        except Exception as e:
            print(f"加载NER模型时出错: {str(e)}")
            raise

    def _get_dimension_transform(self, input_dim, target_dim, transform_cache):
        """获取或创建维度转换层
        
        Args:
            input_dim: 输入维度
            target_dim: 目标维度
            transform_cache: 转换层缓存字典
            
        Returns:
            转换层
        """
        cache_key = f"{input_dim}_to_{target_dim}"
        
        if cache_key not in transform_cache:
            # 创建新的转换层
            transform_layer = torch.nn.Sequential(
                torch.nn.Linear(input_dim, target_dim),
                torch.nn.ReLU(),
                torch.nn.LayerNorm(target_dim)
            ).to(DEVICE)
            # 设置为eval模式
            transform_layer.eval()
            transform_cache[cache_key] = transform_layer
    
        
        return transform_cache[cache_key]

    def _adjust_feature_dimension(self, feature, target_dim, transform_cache, feature_name="特征"):
        """使用线性变换调整特征维度
        
        Args:
            feature: 输入特征向量
            target_dim: 目标维度
            transform_cache: 转换层缓存
            feature_name: 特征名称，用于日志
            
        Returns:
            调整后的特征向量
        """
        input_dim = len(feature)
        
        if input_dim == target_dim:
            return feature
        
        # 转换为tensor
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # 获取或创建转换层
        transform_layer = self._get_dimension_transform(input_dim, target_dim, transform_cache)
        
        # 应用转换
        with torch.no_grad():
            adjusted_feature = transform_layer(feature_tensor)
            
        # 标准化
        adjusted_feature = adjusted_feature / (adjusted_feature.norm() + 1e-8)
        
        return adjusted_feature[0].cpu().numpy()

    def extract_features(self, texts, image_paths_groups):
        """提取所有特征，支持多图融合和模态缺失处理"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        image_features = []
        text_features = []
        ner_features = []
        tags = []

        for text, image_paths in zip(texts, image_paths_groups):
            try:
                # 提取文本特征和NER特征
                text_feature, entity_feature = self._extract_text_features(text)
                
                # 使用线性变换调整特征维度，避免信息损失
                text_feature = self._adjust_feature_dimension(
                    text_feature, DESIRED_DIM, self.text_dim_transform, "文本特征"
                )
                
                entity_feature = self._adjust_feature_dimension(
                    entity_feature, DESIRED_DIM, self.entity_dim_transform, "实体特征"
                )
                
                # 确保特征已标准化
                text_feature = text_feature / (np.linalg.norm(text_feature) + 1e-8)
                entity_feature = entity_feature / (np.linalg.norm(entity_feature) + 1e-8)
                
                # 拼接特征
                combined_text_feature = np.concatenate([text_feature, entity_feature])
                
                # 处理图像特征
                if image_paths and any(os.path.exists(path) for path in image_paths):
                    # 有图片的情况
                    group_image_features = []
                    for image_path in image_paths:
                        if not os.path.exists(image_path):
                            continue
                        
                        # 直接在这里处理图像，不使用封装函数
                        try:
                            pil_image = Image.open(image_path).convert('RGB')
                            processed_image = self._preprocess_image(pil_image)
                            if processed_image is None:
                                continue
                            
                            with torch.no_grad():
                                image_feature = self._extract_image_features(processed_image)
                                if image_feature is not None:
                                    group_image_features.append(image_feature)
                                    
                        except Exception as e:
                            continue
                    
                    if group_image_features:
                        if len(group_image_features) > 1:
                            fused_image_feature = self._fuse_multi_image_features(group_image_features)
                        else:
                            fused_image_feature = group_image_features[0]
                    else:
                        # 图片加载失败，使用文本生成图像特征
                        fused_image_feature = self._generate_image_feature_from_text(text_feature)
                else:
                    # 没有图片的情况，使用文本生成图像特征
                    fused_image_feature = self._generate_image_feature_from_text(text_feature)

                # 添加特征
                image_features.append(fused_image_feature)
                text_features.append(combined_text_feature)

                # 提取hashtags
                hashtags = re.findall(r"#(\w+)", text.lower())
                tags.append(hashtags)

                # 保存特征信息
                ner_features.append({
                    "text": text,
                    "image_paths": image_paths if image_paths else [],
                    "has_image": bool(image_paths and any(os.path.exists(path) for path in image_paths)),
                    "entity_feature": entity_feature.tolist()
                })
                
            except Exception as e:
                continue

            if len(image_features) % 100 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not image_features:
            raise ValueError("没有成功处理任何样本，请检查数据和错误信息")

        return np.stack(image_features), np.stack(text_features), tags, ner_features

    def extract_multiview_features(self, texts, image_paths_groups):
        """
        提取多视角特征用于t-SVD聚类
        
        Args:
            texts: 文本列表
            image_paths_groups: 图像路径组列表
            
        Returns:
            多视角特征字典
        """
        # 提取基础特征
        image_features, text_features, tags, ner_features = self.extract_features(texts, image_paths_groups)
        
        # 分离文本特征和实体特征
        text_only_features = []
        entity_only_features = []
        
        for i, combined_feature in enumerate(text_features):
            # 分离文本特征和实体特征
            text_dim = DESIRED_DIM
            text_only = combined_feature[:text_dim]
            entity_only = combined_feature[text_dim:]
            
            text_only_features.append(text_only)
            entity_only_features.append(entity_only)
        
        # 构建多视角特征
        multiview_features = {
            'text': np.array(text_only_features),
            'image': np.array(image_features),
            'entity': np.array(entity_only_features),
            'ner_info': ner_features,
            'tags': tags
        }
        
        return multiview_features

    def _fuse_multi_image_features(self, image_features):
        """使用注意力机制融合多张图片的特征
        
        Args:
            image_features: 列表，包含多张图片的特征向量
            
        Returns:
            融合后的特征向量
        """
        # 将特征转换为tensor
        features = torch.tensor(np.stack(image_features), dtype=torch.float32).to(DEVICE)
        
        # 计算特征间的相似度矩阵
        similarity = torch.matmul(features, features.t())
        
        # 计算注意力权重
        attention_weights = torch.softmax(similarity / np.sqrt(features.shape[1]), dim=1)
        
        # 加权平均
        fused_feature = torch.matmul(attention_weights, features)
        
        # 取平均得到最终特征
        final_feature = torch.mean(fused_feature, dim=0)
        
        # L2标准化
        final_feature = final_feature / final_feature.norm()
        
        return final_feature.cpu().numpy()

    def _preprocess_image(self, image):
        """预处理图像用于图像编码器"""
        try:
            processed = self.image_processor(images=image, return_tensors="pt")
            
            # 将预处理结果移动到正确的设备上
            if isinstance(processed, dict):
                for key, value in processed.items():
                    if torch.is_tensor(value):
                        processed[key] = value.to(DEVICE)
            elif hasattr(processed, 'pixel_values'):
                if torch.is_tensor(processed.pixel_values):
                    processed.pixel_values = processed.pixel_values.to(DEVICE)
            
            return processed
        except Exception as e:
            return None

    def _extract_image_features(self, processed_image):
        """提取图像特征"""
        try:
            with torch.no_grad():
                # 处理BatchFeature对象或字典
                if hasattr(processed_image, 'pixel_values'):
                    model_inputs = {'pixel_values': processed_image.pixel_values}
                elif isinstance(processed_image, dict):
                    model_inputs = processed_image
                else:
                    raise ValueError(f"不支持的预处理结果类型: {type(processed_image)}")
                
                # 获取图像特征
                outputs = self.image_model(**model_inputs)
                
                # 尝试获取pooler_output
                if hasattr(outputs, 'pooler_output'):
                    image_features = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    # 如果没有pooler_output，使用last_hidden_state的平均池化
                    image_features = outputs.last_hidden_state.mean(dim=1)
                else:
                    # 尝试使用第一个属性
                    first_attr = next((attr for attr in dir(outputs) if not attr.startswith('_') and hasattr(getattr(outputs, attr), 'shape')), None)
                    if first_attr:
                        image_features = getattr(outputs, first_attr)
                    else:
                        raise ValueError("无法从模型输出中获取特征")
                
                # 标准化特征
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # 调整维度到DESIRED_DIM
                feature_dim = image_features.shape[1]
                
                if feature_dim != DESIRED_DIM:
                    image_features = self._adjust_feature_dimension(
                        image_features[0].cpu().numpy(), 
                        DESIRED_DIM, 
                        self.image_dim_transform, 
                        "图像特征"
                    )
                    return image_features
                else:
                    return image_features[0].cpu().numpy()
                    
        except Exception as e:
            # 返回零向量
            return np.zeros(DESIRED_DIM, dtype=np.float32)

    def _extract_text_features(self, text):
        """提取文本特征"""
        try:
            # 确保文本是字符串类型
            if not isinstance(text, str):
                text = str(text)
            
            # 处理特殊字符，保留中文和基本英文字符
            text = re.sub(r'[^\u4e00-\u9fff\w\s,.!?，。！？、]', '', text).strip()
            if not text:
                text = "空文本"
            
            # 如果文本只包含数字，添加一些上下文
            if re.match(r'^[\d\s]+$', text) or text == "空文本":
                text = "这是一段文本：" + text
            
            # 使用文本编码器提取特征
            with torch.no_grad():
                # 对文本进行分词和编码
                inputs = self.text_tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_attention_mask=True
                )
                
                # 将输入移到设备上
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                # 获取文本特征
                outputs = self.text_model(**inputs)
                text_features = outputs.last_hidden_state
                
                # 使用平均池化得到文本表示
                attention_mask = inputs['attention_mask']
                text_features = (text_features * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                
                # 标准化特征
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # 调整维度到DESIRED_DIM
                feature_dim = text_features.shape[1]
                if feature_dim != DESIRED_DIM:
                    text_features = self._adjust_feature_dimension(
                        text_features[0].cpu().numpy(), 
                        DESIRED_DIM, 
                        self.text_dim_transform, 
                        "文本特征"
                    )
                else:
                    text_features = text_features[0].cpu().numpy()
                
        except Exception as e:
            print(f"文本特征提取失败: {str(e)}")
            # 使用字符编码作为备用特征
            char_codes = [ord(c) for c in text[:100]]
            text_features = np.array(char_codes, dtype=np.float32)
            text_features = self._adjust_feature_dimension(
                text_features, DESIRED_DIM, self.text_dim_transform, "字符编码特征"
            )
        
        # 提取NER特征
        try:
            entity_feature = self._extract_ner_features(text)
        except Exception as ner_error:
            print(f"NER特征提取失败: {str(ner_error)}")
            entity_feature = np.zeros(DESIRED_DIM, dtype=np.float32)
        
        return text_features, entity_feature

    def _extract_ner_features(self, text):
        """提取NER特征"""
        try:
            # 对文本进行分词和编码
            inputs = self.ner_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTHS[DATASET_TYPE],
                return_offsets_mapping=True
            )
            
            # 获取token到原始文本的映射
            offset_mapping = inputs.pop("offset_mapping").detach().cpu().numpy()[0]
            
            # 将输入移到设备上
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # 获取模型输出
            with torch.no_grad():
                outputs = self.ner_model(**inputs)
                
            # 获取每个token的预测标签
            predictions = torch.argmax(outputs.logits, dim=-1)[0].detach().cpu().numpy()
            hidden_states = outputs.hidden_states[-1][0].detach()
            
            # 提取实体特征
            entity_features = []
            current_entity = []
            
            for idx, (pred, hidden) in enumerate(zip(predictions, hidden_states)):
                if pred > 0:  # 非O标签
                    current_entity.append(hidden)
                elif current_entity:  # 实体结束
                    # 使用平均池化合并实体token的特征
                    entity_feature = torch.stack(current_entity).mean(dim=0)
                    entity_features.append(entity_feature)
                    current_entity = []
            
            # 处理最后一个实体
            if current_entity:
                entity_feature = torch.stack(current_entity).mean(dim=0)
                entity_features.append(entity_feature)
            
            # 如果找到了实体
            if entity_features:
                # 在GPU上进行特征处理
                entity_features = torch.stack(entity_features).mean(dim=0)
                
                # 使用动态转换层将特征维度调整为DESIRED_DIM
                input_dim = entity_features.shape[0]
                if input_dim != self.ner_model.config.hidden_size:
                    # 使用动态转换层处理不匹配的维度
                    temp_transform = self._get_dimension_transform(
                        input_dim, DESIRED_DIM, self.entity_dim_transform
                    )
                    entity_features = temp_transform(entity_features.unsqueeze(0))[0]
                else:
                    # 使用预定义的转换层
                    entity_features = self.ner_transform(entity_features)
                
                # 标准化特征
                entity_features = entity_features / (entity_features.norm() + 1e-8)
                
                # 转移到CPU并转换为numpy数组
                final_entity_feature = entity_features.detach().cpu().numpy()
                return final_entity_feature
            
            # 如果没有找到实体，返回零向量
            return np.zeros(DESIRED_DIM, dtype=np.float32)
            
        except Exception as ner_error:
            print(f"NER特征提取出错: {str(ner_error)}")
            return np.zeros(DESIRED_DIM, dtype=np.float32)

    def _generate_image_feature_from_text(self, text_feature):
        """使用文本特征生成伪图像特征
        
        Args:
            text_feature: 文本特征向量
            
        Returns:
            生成的图像特征向量
        """
        # 转换为tensor
        text_tensor = torch.tensor(text_feature, dtype=torch.float32).to(DEVICE)
        
        # 确保维度正确
        if len(text_tensor.shape) == 1:
            text_tensor = text_tensor.unsqueeze(0)
            
        # 生成伪图像特征
        with torch.no_grad():
            pseudo_image_feature = self.modality_transform(text_tensor)
            
        # L2标准化
        pseudo_image_feature = pseudo_image_feature / pseudo_image_feature.norm(dim=-1, keepdim=True)
        
        return pseudo_image_feature[0].cpu().numpy() 

def init_feature_extractor():
    global _FEATURE_EXTRACTOR_SINGLETON
    if _FEATURE_EXTRACTOR_SINGLETON is None:
        _FEATURE_EXTRACTOR_SINGLETON = FeatureExtractor()
    return _FEATURE_EXTRACTOR_SINGLETON

def get_feature_extractor():
    global _FEATURE_EXTRACTOR_SINGLETON
    if _FEATURE_EXTRACTOR_SINGLETON is None:
        raise RuntimeError("FeatureExtractor未初始化，请先调用 init_feature_extractor().")
    return _FEATURE_EXTRACTOR_SINGLETON

def reset_feature_extractor():
    global _FEATURE_EXTRACTOR_SINGLETON
    _FEATURE_EXTRACTOR_SINGLETON = None