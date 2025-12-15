"""
批量聚类脚本
遍历指定目录下的每个子目录，对每个子目录运行聚类程序，并计算评价指标的平均值
"""

import os
import numpy as np
from main import forward, init_feature_extractor
from utils import print_clustering_metrics
import json
from datetime import datetime

def batch_clustering(base_dir, use_saved=False, method='traditional', min_posts=1, 
                    source_site="新浪微博", use_prior=True, max_samples_per_event=1200, 
                    min_samples_per_event=1):
    """
    批量运行聚类程序
    
    Args:
        base_dir: 包含多个子目录的基础目录路径（例如：event_新浪微博_split）
        use_saved: 是否使用已保存的特征
        method: 聚类方法 ('traditional' 或 'tensor')
        min_posts: 最小帖子数量
        source_site: 来源网站过滤条件
        use_prior: 是否使用先验分布
        max_samples_per_event: 每个事件的最大样本数
        min_samples_per_event: 每个事件的最小样本数
    
    Returns:
        dict: 包含所有子目录的聚类结果和平均指标
    """
    # 检查基础目录是否存在
    if not os.path.exists(base_dir):
        print(f"错误：目录不存在: {base_dir}")
        return None
    
    # 获取所有子目录
    subdirs = [d for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d))]
    
    if not subdirs:
        print(f"错误：在 {base_dir} 中未找到任何子目录")
        return None
    
    print(f"找到 {len(subdirs)} 个子目录，开始批量聚类...")
    print("=" * 80)
    
    # 初始化特征提取器（只需要初始化一次）
    print("初始化特征提取器...")
    init_feature_extractor()
    print("特征提取器初始化完成\n")
    
    # 存储所有结果
    all_results = []
    all_metrics = []
    
    # 遍历每个子目录
    for idx, subdir in enumerate(sorted(subdirs), 1):
        data_source_path = os.path.join(base_dir, subdir)
        
        print(f"\n{'=' * 80}")
        print(f"处理子目录 {idx}/{len(subdirs)}: {subdir}")
        print(f"数据路径: {data_source_path}")
        print(f"{'=' * 80}\n")
        
        try:
            # 运行聚类
            result = forward(
                data_source_path=data_source_path,
                use_saved=use_saved,
                method=method,
                min_posts=min_posts,
                source_site=source_site,
                use_prior=use_prior,
                max_samples_per_event=max_samples_per_event,
                min_samples_per_event=min_samples_per_event
            )
            
            if result is None:
                print(f"警告：子目录 {subdir} 的聚类失败，跳过")
                continue
            
            # 提取指标
            metrics = result.get('metrics', {})
            
            # 保存结果
            subdir_result = {
                'subdir': subdir,
                'data_source_path': data_source_path,
                'metrics': metrics,
                'n_clusters': metrics.get('聚类数量', 0),
                'n_samples': metrics.get('样本数量', 0)
            }
            all_results.append(subdir_result)
            all_metrics.append(metrics)
            
            # 打印当前子目录的结果
            print(f"\n子目录 {subdir} 的聚类结果:")
            print_clustering_metrics(metrics)
            
        except Exception as e:
            print(f"错误：处理子目录 {subdir} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 计算平均指标
    if not all_metrics:
        print("\n错误：没有成功完成任何聚类任务")
        return None
    
    print(f"\n{'=' * 80}")
    print("批量聚类完成！")
    print(f"{'=' * 80}\n")
    
    # 计算三种主要指标的平均值
    purity_values = [m.get('纯度', 0) for m in all_metrics if '纯度' in m]
    nmi_values = [m.get('NMI', 0) for m in all_metrics if 'NMI' in m]
    ari_values = [m.get('ARI', 0) for m in all_metrics if 'ARI' in m]
    
    avg_metrics = {
        '纯度': np.mean(purity_values) if purity_values else 0,
        'NMI': np.mean(nmi_values) if nmi_values else 0,
        'ARI': np.mean(ari_values) if ari_values else 0,
        '聚类数量': np.mean([m.get('聚类数量', 0) for m in all_metrics]),
        '样本数量': np.sum([m.get('样本数量', 0) for m in all_metrics])
    }
    
    # 打印平均指标
    print("=" * 80)
    print("平均评价指标（基于所有子目录）:")
    print("=" * 80)
    print_clustering_metrics(avg_metrics)
    print(f"平均聚类数量: {avg_metrics['聚类数量']:.2f}")
    print(f"总样本数量: {avg_metrics['样本数量']}")
    print("=" * 80)
    
    # 打印每个子目录的详细结果
    print("\n各子目录详细结果:")
    print("-" * 80)
    for result in all_results:
        print(f"\n子目录: {result['subdir']}")
        print(f"  纯度: {result['metrics'].get('纯度', 0):.4f}")
        print(f"  NMI: {result['metrics'].get('NMI', 0):.4f}")
        print(f"  ARI: {result['metrics'].get('ARI', 0):.4f}")
        print(f"  聚类数量: {result['n_clusters']}")
        print(f"  样本数量: {result['n_samples']}")
    
    # 保存结果到JSON文件
    output_file = f"batch_clustering_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_data = {
        'base_dir': base_dir,
        'method': method,
        'total_subdirs': len(subdirs),
        'successful_runs': len(all_results),
        'average_metrics': avg_metrics,
        'detailed_results': all_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    return {
        'average_metrics': avg_metrics,
        'detailed_results': all_results,
        'output_file': output_file
    }

def main():
    """主函数"""
    # 配置参数
    base_dir = "event_新浪微博_split"  # 包含多个子目录的基础目录
    use_saved = False  # 是否使用已保存的特征
    method = 'traditional'  # 聚类方法: 'traditional' 或 'tensor'
    min_posts = 1  # 最小帖子数量
    source_site = "新浪微博"  # 来源网站过滤条件
    use_prior = True  # 是否使用先验分布
    max_samples_per_event = 1200  # 每个事件的最大样本数
    min_samples_per_event = 1  # 每个事件的最小样本数
    
    # 运行批量聚类
    result = batch_clustering(
        base_dir=base_dir,
        use_saved=use_saved,
        method=method,
        min_posts=min_posts,
        source_site=source_site,
        use_prior=use_prior,
        max_samples_per_event=max_samples_per_event,
        min_samples_per_event=min_samples_per_event
    )
    
    if result is None:
        print("批量聚类失败")
        return 1
    
    return 0

if __name__ == "__main__":
    main()

