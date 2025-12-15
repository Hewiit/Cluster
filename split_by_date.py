import os
import glob
import pandas as pd
from datetime import datetime, timedelta


def parse_date_series(date_series):
    """
    将日期列解析为 pandas.Timestamp，自动处理 'YYYY/M/D HH:MM' 形式。
    """
    return pd.to_datetime(date_series, errors="coerce")


def get_global_date_range(platform_dir, date_col="日期", encoding="utf-8-sig"):
    """
    扫描平台目录下所有 csv，得到全局最早和最晚日期（按日期部分）。
    """
    min_date = None
    max_date = None

    for subdir_name in os.listdir(platform_dir):
        subdir_path = os.path.join(platform_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            continue

        csv_files = glob.glob(os.path.join(subdir_path, "*.csv"))
        if not csv_files:
            continue

        csv_path = csv_files[0]
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
        except Exception as e:
            print(f"读取失败，跳过：{csv_path}，错误：{e}")
            continue

        if date_col not in df.columns:
            print(f"找不到日期列 '{date_col}'，跳过：{csv_path}")
            continue

        dates = parse_date_series(df[date_col]).dropna()
        if dates.empty:
            continue

        # 只取日期部分
        local_min = dates.dt.date.min()
        local_max = dates.dt.date.max()

        if min_date is None or local_min < min_date:
            min_date = local_min
        if max_date is None or local_max > max_date:
            max_date = local_max

    return min_date, max_date


def generate_date_windows(start_date, end_date, window_days=7):
    """
    生成从 start_date 到 end_date 的多个时间窗口，[start, start+6]，最后一个窗口可能不足 7 天。
    """
    windows = []
    current = start_date
    delta = timedelta(days=window_days - 1)  # 如 7 天窗口，即 +6

    while current <= end_date:
        win_start = current
        win_end = min(current + delta, end_date)
        windows.append((win_start, win_end))
        current = win_end + timedelta(days=1)

    return windows


def split_platform_by_date(
    platform_dir,
    source_name,
    date_col="日期",
    encoding="utf-8-sig",
    window_days=7,
):
    """
    按日期把某个来源网站的平台目录划分成多个 event_wh_{来源网站}_{开始日期}_{结束日期} 目录。
    所有分好的目录都会放到 event_wh_{来源网站}_split 这个大目录中。
    目录结构保持一致，csv 文件名追加 _{开始日期}_{结束日期} 后缀。
    """
    platform_dir = os.path.abspath(platform_dir)
    if not os.path.isdir(platform_dir):
        raise FileNotFoundError(f"目录不存在：{platform_dir}")

    parent_dir = os.path.dirname(platform_dir)

    # 创建总的大目录，用于存放所有分好的目录
    output_base_dir = os.path.join(parent_dir, f"event_{source_name}_split")
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"输出目录：{output_base_dir}")

    # 1. 全局日期范围
    global_min, global_max = get_global_date_range(platform_dir, date_col, encoding)
    if global_min is None or global_max is None:
        print("未能从数据中解析出有效日期，程序结束。")
        return

    print(f"全局日期范围：{global_min} ~ {global_max}")

    # 2. 生成时间窗口
    windows = generate_date_windows(global_min, global_max, window_days)
    print(f"共生成 {len(windows)} 个时间窗口。")

    # 3. 对每个子目录、每个窗口进行划分
    for subdir_name in os.listdir(platform_dir):
        subdir_path = os.path.join(platform_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            continue

        csv_files = glob.glob(os.path.join(subdir_path, "*.csv"))
        if not csv_files:
            continue

        csv_path = csv_files[0]
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
        except Exception as e:
            print(f"读取失败，跳过：{csv_path}，错误：{e}")
            continue

        if date_col not in df.columns:
            print(f"找不到日期列 '{date_col}'，跳过：{csv_path}")
            continue

        dates = parse_date_series(df[date_col])
        df["_parsed_date"] = dates.dt.date  # 只要日期部分

        # 源文件名（不含路径）
        base_name = os.path.basename(csv_path)
        name_no_ext, ext = os.path.splitext(base_name)

        for win_start, win_end in windows:
            mask = (df["_parsed_date"] >= win_start) & (df["_parsed_date"] <= win_end)
            sub_df = df[mask].drop(columns=["_parsed_date"])
            if sub_df.empty:
                continue

            # 目录名和文件名中的日期格式：YYYYMMDD
            start_str = win_start.strftime("%Y%m%d")
            end_str = win_end.strftime("%Y%m%d")

            # 目标根目录名：event_wh_{来源网站}_{开始日期}_{结束日期}
            target_root_name = f"event_wh_{source_name}_{start_str}_{end_str}"
            target_root = os.path.join(output_base_dir, target_root_name)

            # 目标子目录与原来一致
            target_subdir = os.path.join(target_root, subdir_name)
            os.makedirs(target_subdir, exist_ok=True)

            # 新文件名：原文件名 + _{开始日期}_{结束日期} + .csv
            target_filename = f"{name_no_ext}_{start_str}_{end_str}{ext}"
            target_path = os.path.join(target_subdir, target_filename)

            try:
                sub_df.to_csv(target_path, index=False, encoding=encoding)
                print(
                    f"写出：{target_path}（{len(sub_df)} 行，{win_start} ~ {win_end}）"
                )
            except Exception as e:
                print(f"写出失败：{target_path}，错误：{e}")


if __name__ == "__main__":
    # 示例：对新浪微博目录做划分
    # 平台目录：event_wh_新浪微博
    # 来源网站名称（用于目录名中）：新浪微博
    split_platform_by_date(
        platform_dir="events_新浪微博",
        source_name="新浪微博",
        date_col="日期",
        encoding="utf-8-sig",
        window_days=1,
    )