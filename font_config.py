"""
matplotlib中文字体配置文件
解决t-SNE可视化中中文显示乱码的问题
"""

import matplotlib
import matplotlib.pyplot as plt
import os
import platform

def configure_chinese_fonts():
    """
    配置matplotlib支持中文字体显示
    
    Returns:
        bool: 配置是否成功
    """
    try:
        # 基础字体配置
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['font.size'] = 12
        
        # 根据操作系统选择合适的字体
        if platform.system() == 'Darwin':  # macOS
            font_list = [
                'PingFang SC',           # macOS系统字体
                'Hiragino Sans GB',      # macOS系统字体
                'STHeiti',               # macOS系统字体
                'SimHei',                # 黑体
                'DejaVu Sans'            # 备用字体
            ]
        elif platform.system() == 'Linux':
            font_list = [
                'WenQuanYi Micro Hei',  # Linux中文字体
                'Noto Sans CJK SC',     # Google Noto字体
                'SimHei',                # 黑体
                'DejaVu Sans'            # 备用字体
            ]
        elif platform.system() == 'Windows':
            font_list = [
                'SimHei',                # 黑体
                'Microsoft YaHei',       # 微软雅黑
                'SimSun',                # 宋体
                'DejaVu Sans'            # 备用字体
            ]
        else:
            # 其他系统使用通用字体
            font_list = [
                'SimHei',                # 黑体
                'DejaVu Sans',           # 备用字体
                'Arial Unicode MS'       # Unicode字体
            ]
        
        # 设置字体列表
        matplotlib.rcParams['font.sans-serif'] = font_list
        
        # 同步到plt
        plt.rcParams['font.sans-serif'] = font_list
        
        # 测试字体是否可用
        test_fig, test_ax = plt.subplots(figsize=(1, 1))
        test_ax.text(0.5, 0.5, '测试中文', fontsize=12)
        plt.close(test_fig)
        
        print("✅ 中文字体配置成功")
        print(f"使用的字体: {font_list[0]}")
        return True
        
    except Exception as e:
        print(f"❌ 字体配置失败: {e}")
        print("将使用默认字体，中文可能显示为方块")
        
        # 设置备用配置
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        return False

def get_available_fonts():
    """
    获取系统中可用的字体列表
    
    Returns:
        list: 可用字体列表
    """
    try:
        from matplotlib.font_manager import FontManager
        fm = FontManager()
        font_names = [f.name for f in fm.ttflist]
        return sorted(set(font_names))
    except Exception as e:
        print(f"无法获取字体列表: {e}")
        return []

def find_chinese_fonts():
    """
    查找系统中的中文字体
    
    Returns:
        list: 中文字体列表
    """
    available_fonts = get_available_fonts()
    chinese_fonts = []
    
    # 常见的中文字体关键词
    chinese_keywords = [
        'SimHei', 'SimSun', 'Microsoft YaHei', 'PingFang', 'Hiragino',
        'STHeiti', 'WenQuanYi', 'Noto', 'Source Han', '思源', '黑体', '宋体'
    ]
    
    for font in available_fonts:
        for keyword in chinese_keywords:
            if keyword.lower() in font.lower():
                chinese_fonts.append(font)
                break
    
    return sorted(set(chinese_fonts))

if __name__ == "__main__":
    print("=== matplotlib中文字体配置工具 ===")
    
    # 配置字体
    success = configure_chinese_fonts()
    
    if success:
        print("\n✅ 字体配置完成")
    else:
        print("\n❌ 字体配置失败，尝试查找可用字体...")
        
        # 查找可用字体
        chinese_fonts = find_chinese_fonts()
        if chinese_fonts:
            print(f"发现 {len(chinese_fonts)} 个中文字体:")
            for font in chinese_fonts[:10]:  # 只显示前10个
                print(f"  - {font}")
            if len(chinese_fonts) > 10:
                print(f"  ... 还有 {len(chinese_fonts) - 10} 个字体")
        else:
            print("未发现中文字体")
    
    print("\n=== 配置完成 ===")
