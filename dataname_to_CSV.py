import os
import csv

# 指定要遍歷的資料夾路徑
folder_path = "/home/anywhere3090l/Desktop/DDAD-main/thyroid/thyroid/test/ground_truth/"

# 設定 CSV 檔案的名稱
csv_file = "file_list.csv"

# 遍歷資料夾並取得所有檔案名稱
file_names = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_names.append(file)

# 寫入檔案名稱到 CSV 檔案
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["File Name"])  # 寫入 CSV 檔案的欄位名稱
    for name in file_names:
        writer.writerow([name])

print("檔案名稱已經儲存到", csv_file)