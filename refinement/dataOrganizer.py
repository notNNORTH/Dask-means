import csv
import os
import shutil

import pandas as pd
import re

data_path = "/mnt/e/torch-clus/logs/icde-fmeans/"
data_destination = "/mnt/e/dask-means/dataset"


def organizer(data_destination="/mnt/e/dask-means/dataset-series/"):
    # 确保目标路径存在
    os.makedirs(data_destination, exist_ok=True)

    cnt = 0
    # 遍历数据源路径下的所有文件
    for filename in os.listdir(data_path):
        # if cnt > 1:
        #     break
        # cnt += 1

        L = filename.split("-")
        d = "2" if "map" in filename else "3"
        k = re.findall(r'\d+', L[0])[0]
        n = L[1]
        c = L[2]
        t = L[3]

        # print(d, k, n, c, t)
        if filename.endswith(".csv"):
            file_path = os.path.join(data_path, filename)
            try:
                data = pd.read_csv(file_path)
                print(f"Processing file: {filename}")
                last_row = data.iloc[-1]
                row_sum = list(last_row[last_row.apply(lambda x: isinstance(x, (int, float)))])
                # print(row_sum)
                output_filename = filename  # .replace('.csv', '.txt')
                output_file_path = os.path.join(data_destination, output_filename)
                with open(output_file_path, 'w') as output_file:
                    output_line = f"{d},{k},{n},{c},{t}," + ",".join(map(str, row_sum)) + "\n"
                    output_file.write(output_line)
                    # output_file.write("%s,%s,%s,%s,%s" % (d, k, n, c, t), row_sum)
                    print()
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print("Processing complete. Results are written to", data_destination)


def combine(folder_path="/mnt/e/dask-means/dataset-series", output_file_path="/mnt/e/dask-means/kmeans-time-series.csv"):
    # 确定文件夹中所有的 CSV 文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 打开输出文件，准备写入
    with open(output_file_path, 'w', newline='') as file_output:
        writer = csv.writer(file_output)

        # 遍历文件列表，将每个文件的内容转换成一个字符串，然后作为一行写入
        for file in csv_files:
            file_path = os.path.join(folder_path, file)

            # 读取每个 CSV 文件
            with open(file_path, newline='') as file_input:
                reader = csv.reader(file_input)
                # 将所有行的内容合并为一个单独的列表
                all_rows = [cell for row in reader for cell in row]
                # 写入单行到输出文件
                writer.writerow(all_rows)

    print(f"All CSV files have been combined into {output_file_path}")


def rename(folder_path="/mnt/e/dask-means/dataset", output_file_path="/mnt/e/dask-means/dataset sorted by n/"):
    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 构建完整的文件路径
        original_file = os.path.join(folder_path, filename)

        # 检查是否为文件（非目录）
        if os.path.isfile(original_file):
            new_filename = filename.split("-")[1] + ".csv"
            print(new_filename)
            new_file_path = os.path.join(output_file_path, new_filename)

            # 复制文件到新位置，并重命名
            shutil.copy2(original_file, new_file_path)


if __name__ == '__main__':
    print("Start organizing data!")
    # organizer()
    combine()
    # rename()