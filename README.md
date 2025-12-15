# 多模态聚类项目

## 项目简介

本项目是一个**非训练项目**，用于对多模态数据（文本和图像）进行聚类分析。项目直接加载HuggingFace预训练模型或本地预训练模型进行特征提取，然后使用聚类算法对数据进行分组。

### 模型加载方式

1. **本地模型**（优先）：项目会优先从 `./models/` 目录加载模型，可直接下载[预训练模型](https://drive.google.com/drive/folders/1ieJurPVGWEzBGoIXBV-ZqMTmOBZjvcw8?usp=drive_link)。
   - 如果模型存在于本地，直接使用本地模型
   - 本地模型目录结构：
     ```
     models/
     ├── bert-base-chinese/
     ├── chinese-bert-wwm-ext/
     └── clip-vit-base-patch32/
     ```

2. **HuggingFace模型**（备选）：如果本地模型不存在，会自动从HuggingFace下载
   - 首次运行时，模型会自动下载并缓存
   - 后续运行时会使用缓存模型

## 数据格式

### 多模态数据集格式

目录结构：
```
event_wh/
├── 事件1/
│   ├── 事件1.csv          # 包含id和微博正文字段
│   └── images/            # 图片目录（可选）
│       ├── {id}.jpg
│       └── {id}-0.jpg
└── 事件2/
    └── ...
```

CSV文件必需字段：
- `id`：数据唯一标识
- `微博正文`：文本内容

### 纯文本数据集格式

目录结构：
```
event_wh/
├── 事件1/
│   └── *.csv              # CSV文件，包含多平台数据
└── 事件2/
    └── ...
```

CSV文件主要字段：
- `原微博内容` 或 `全文内容`：文本内容
- `来源网站`：数据来源平台
- `日期`：发布日期
- 其他字段：转发数、评论数、点赞数等


## 使用方法

### 基本使用



1. **运行主程序**：
   ```python
   python main.py
   ```

2. **主要参数配置**（在 `main.py` 的 `main()` 函数中）：
   ```python
   data_source_path = 'wvent_wh'      # 可选，使用默认路径时可设为None
   use_saved = False                  # 是否使用已保存的特征
   source_site = "新浪微博"           # 纯文本数据集的来源网站过滤条件
   ```


## 结果保存形式

### 聚类结果保存

聚类结果保存在 `cluster_events/` 目录下，结构如下：

```
cluster_events/
├── 事件_0/                    # 簇0对应的事件目录
│   └── 事件名称.csv           # 包含该簇的所有数据
├── 事件_1/                    # 簇1对应的事件目录
│   └── 事件名称.csv
└── ...
```

每个CSV文件包含：
- **原始数据的所有字段**：保留原始数据的所有列
- **簇ID**：`簇ID` 列，标识该数据所属的簇
- **事件名称**：`事件名称` 列

### 特征保存

提取的特征保存在 `extracted_features/` 目录下：
- `image_features.npy`：图像特征
- `text_features.npy`：文本特征
- `ner_features.json`：NER实体特征
- `true_labels.npy`：真实标签（如果存在）

### 特征复用

设置 `use_saved=True` 可以复用已提取的特征，避免重复提取：
- 如果特征已存在且有效，直接加载使用
- 如果特征不存在或需要重新提取，设置为 `False`

## 项目结构

```
muitiCluster/
├── main.py                 # 主程序入口
├── config.py              # 配置文件
├── data_loader.py         # 数据加载模块
├── feature_extractor.py   # 特征提取模块（加载预训练模型）
├── clustering.py          # 传统聚类方法
├── tensor_clustering.py   # 张量聚类方法
├── utils.py               # 工具函数（保存结果等）
├── split_by_date.py       # 按日期分割数据工具
├── splitcsv.py            # CSV分割工具
├── models/                # 本地模型目录（可选）
│   ├── bert-base-chinese/
│   └── ...
├── extracted_features/    # 提取的特征（自动生成）
├── cluster_events/        # 聚类结果（自动生成）
└── README.md             # 本文件
```

## 依赖环境

主要依赖包：
- `torch`：PyTorch深度学习框架
- `transformers`：HuggingFace transformers库
- `numpy`：数值计算
- `pandas`：数据处理
- `scikit-learn`：聚类算法
- `PIL`：图像处理

