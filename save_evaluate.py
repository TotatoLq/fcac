import os
import csv


def save_evaluating_indicator (
        result,
        data_name,
        ARI,
        NMI,
        ACC,
        save_dir="./acc_records/fcac"
    ):
    """
    创建或追加写入 acc_xxx.csv 文件
    文件格式：
        num_rounds, val_acc, test_acc
    """

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 文件名
    file_name = f"{result}_{data_name}.csv"
    file_path = os.path.join(save_dir, file_name)

    # 判断文件是否存在
    file_exists = os.path.isfile(file_path)

    # 如果不存在，新建并写入表头
    if not file_exists:
        with open(file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ "ARI","NMI",  "ACC"])
        print(f"[INFO] 创建新文件：{file_path}")

    # 追加写入一行数据
    with open(file_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ ARI,NMI, ACC])
