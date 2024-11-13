import csv
import json

json_file_path = '../../../VQA_Red_data/VQA_RAD Dataset Public.json'

# 读取JSON文件
try:
    with open(json_file_path, 'r') as file:
        data = json.load(file)
except Exception as e:
    print(f"Error reading JSON file: {e}")

# 创建一个字典来保存每个image_name及其对应的唯一ID
image_ids = {}
for item in data:
    if item["image_name"] not in image_ids:
        image_ids[item["image_name"]] = len(image_ids) + 1  # 分配一个唯一的ID

# 将image_name替换为对应的ID，并保留原始的image_name
for item in data:
    original_image_name = item["image_name"]
    item["image_id"] = image_ids[original_image_name]
    item["original_image_name"] = original_image_name

# 根据图片ID排序数据
sorted_data = sorted(data, key=lambda x: x["image_id"])

# 写入CSV文件
with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ["original_image_name", "image_id", "question", "answer"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for item in sorted_data:
        # 只保留所选字段
        filtered_item = {k: item[k] for k in fieldnames}
        writer.writerow(filtered_item)

print("CSV file created successfully.")