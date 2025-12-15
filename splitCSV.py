import os
import glob
import pandas as pd

def split_event_wh_by_source(
    base_dir="event_wh",
    source_col="来源网站",
    encoding="utf-8-sig"
):
    """
    根据来源网站字段，将 event_wh 目录划分成多个 event_wh_{来源网站} 目录。
    每个新目录下的结构与原 event_wh 相同，但 csv 文件名改为 子目录名_来源网站.csv。
    """
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"目录不存在：{base_dir}")

    parent_dir = os.path.dirname(base_dir)

    # 遍历 event_wh 下的所有子目录
    for subdir_name in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            continue

        # 寻找这个子目录下唯一的 csv 文件
        csv_files = glob.glob(os.path.join(subdir_path, "*.csv"))
        if not csv_files:
            continue
        csv_path = csv_files[0]

        try:
            df = pd.read_csv(csv_path, encoding=encoding)
        except Exception as e:
            print(f"读取失败，跳过：{csv_path}，错误：{e}")
            continue

        if source_col not in df.columns:
            print(f"找不到列 '{source_col}'，跳过：{csv_path}")
            continue

        # 去掉来源网站为空的行
        df = df.dropna(subset=[source_col])

        # 按来源网站分组
        for source, group in df.groupby(source_col):
            if not isinstance(source, str) or source.strip() == "":
                continue

            source_str = str(source).strip()

            # 目标根目录 event_wh_来源网站
            target_root = os.path.join(parent_dir, f"event_wh_{source_str}")
            # 目标子目录名与原子目录相同
            target_subdir = os.path.join(target_root, subdir_name)
            os.makedirs(target_subdir, exist_ok=True)

            # 新文件名：子目录名_来源网站.csv
            # 也可以按你需要自定义命名规则
            safe_source = source_str.replace("/", "_").replace("\\", "_")
            target_filename = f"{subdir_name}_{safe_source}.csv"
            target_path = os.path.join(target_subdir, target_filename)

            try:
                group.to_csv(target_path, index=False, encoding=encoding)
                print(f"写出：{target_path}（{len(group)} 行，平台：{source_str}）")
            except Exception as e:
                print(f"写出失败：{target_path}，错误：{e}")


if __name__ == "__main__":
    # 在项目根目录下执行：python split_event_wh_by_source.py
    split_event_wh_by_source()