#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版微博事件目录过滤器
读取微博帖子目录中的事件子目录，统计每个CSV文件的帖子数
将帖子数少于10条的事件目录移动到新的目录中
"""

import os
import csv
import shutil
from pathlib import Path

def count_posts_in_csv(csv_file_path):
    """统计CSV文件中的帖子数（不包括表头）"""
    try:
        with open(csv_file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = list(reader)
            # 减去表头，如果没有表头则返回0
            return max(0, len(rows) - 1) if rows else 0
    except Exception as e:
        print(f"读取CSV文件失败 {csv_file_path}: {e}")
        return 0

def filter_events(base_dir="结果文件", min_posts=10):
    """
    过滤事件目录
    
    Args:
        base_dir (str): 基础目录路径
        min_posts (int): 最小帖子数阈值
    """
    base_path = Path(base_dir)
    
    # 检查基础目录是否存在
    if not base_path.exists() or not base_path.is_dir():
        print(f"错误：目录不存在或不是目录: {base_dir}")
        return
    
    # 创建过滤目录
    filtered_dir = base_path / f"帖子数少于{min_posts}条事件"
    if not filtered_dir.exists():
        filtered_dir.mkdir(parents=True)
        print(f"创建过滤目录: {filtered_dir}")
    
    # 获取所有子目录
    event_dirs = [d for d in base_path.iterdir() if d.is_dir() and d != filtered_dir]
    
    if not event_dirs:
        print(f"在 {base_dir} 中没有找到事件子目录")
        return
    
    print(f"开始处理 {len(event_dirs)} 个事件目录...")
    print(f"最小帖子数阈值: {min_posts}")
    print("-" * 60)
    
    total_events = 0
    moved_events = 0
    total_posts = 0
    event_details = []
    
    # 处理每个事件目录
    for event_dir in event_dirs:
        event_name = event_dir.name
        total_events += 1
        
        # 查找CSV文件
        csv_files = list(event_dir.glob("*.csv"))
        
        if not csv_files:
            print(f"警告：{event_name} 目录中没有找到CSV文件")
            event_details.append([event_name, 0, "无CSV文件", "未移动"])
            continue
        
        # 使用第一个CSV文件
        csv_file = csv_files[0]
        post_count = count_posts_in_csv(csv_file)
        total_posts += post_count
        
        print(f"事件: {event_name:<20} | 帖子数: {post_count:>3} | CSV文件: {csv_file.name}")
        
        # 判断是否需要移动
        if post_count < min_posts:
            moved_events += 1
            target_path = filtered_dir / event_name
            
            try:
                # 如果目标路径已存在，先删除
                if target_path.exists():
                    shutil.rmtree(target_path)
                
                # 移动目录
                shutil.move(str(event_dir), str(target_path))
                print(f"  ✓ 已移动到: {target_path}")
                event_details.append([event_name, post_count, "少于阈值", "已移动"])
                
            except Exception as e:
                print(f"  ✗ 移动失败: {e}")
                event_details.append([event_name, post_count, "少于阈值", "移动失败"])
        else:
            print(f"  - 保留在原位置")
            event_details.append([event_name, post_count, "达到阈值", "保留"])
    
    # 打印统计结果
    print("\n" + "=" * 60)
    print("过滤完成！统计摘要:")
    print(f"总事件数: {total_events}")
    print(f"移动的事件数: {moved_events}")
    print(f"保留的事件数: {total_events - moved_events}")
    print(f"总帖子数: {total_posts}")
    print(f"过滤目录: {filtered_dir}")
    print("=" * 60)
    
    # 保存详细统计结果
    save_statistics(event_details, "event_statistics.csv")
    
    return event_details

def save_statistics(event_details, output_file):
    """保存统计结果到CSV文件"""
    try:
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['事件名称', '帖子数量', '状态说明', '处理结果'])
            
            for detail in event_details:
                writer.writerow(detail)
        
        print(f"\n统计结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"保存统计结果失败: {e}")

def main():
    """主函数"""
    print("微博事件目录过滤器")
    print("=" * 60)
    
    # 配置参数
    base_directory = "结果文件"  # 可以根据需要修改
    min_posts_threshold = 10    # 最小帖子数阈值
    
    # 执行过滤
    filter_events(base_directory, min_posts_threshold)
    
    print("\n程序执行完成！")

if __name__ == "__main__":
    main()
