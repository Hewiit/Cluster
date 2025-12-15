# 多模态聚类分析程序

## 快速开始

### 运行程序

```bash
python main.py
```

### 主要参数配置

在 `main()` 函数中修改以下参数：

- **data_source_path**: 数据源路径（可选，如果为None则使用config中的默认路径）
- **min_posts**: 最小帖子数量（用于数据过滤）
- **source_site**: 来源网站过滤条件（如 "新浪微博"）

### 先验分布参数

在 `forward()` 函数中可配置：

- **use_prior**: 是否使用先验分布（默认True）
- **max_samples_per_event**: 每个事件的最大样本数（默认1000）
- **min_samples_per_event**: 每个事件的最小样本数（默认1）

### 输出结果

聚类结果将保存到 `cluster_events` 目录中，每个簇作为一个事件目录，包含对应的CSV文件。

