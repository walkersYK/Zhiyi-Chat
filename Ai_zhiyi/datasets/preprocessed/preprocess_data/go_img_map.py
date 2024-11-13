import csv
import os


def save_sorted_unique_data_to_txt(csv_file_path, directory='extracted_data'):
    # 创建目录如果它不存在
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 输出文本文件路径
    output_path = os.path.join(directory, 'img_vgg_IDMap.txt')

    # 初始化文件（清空旧内容）
    open(output_path, 'w').close()

    # 创建一个集合来存储唯一的 (original_image_name, image_id) 对
    unique_records = set()

    # 打开 CSV 文件并读取数据
    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        # 遍历每一行数据
        for row in reader:
            original_image_name = row['original_image_name']
            image_id = row['image_id']

            # 将 (original_image_name, image_id) 对添加到集合中
            unique_records.add((original_image_name, image_id))

    # 将集合转换为列表，并按 image_id 排序
    sorted_records = sorted(list(unique_records), key=lambda x: int(x[1]))

    # 将排序后的记录写入文本文件
    with open(output_path, mode='w', encoding='utf-8') as output_file:
        for original_image_name, image_id in sorted_records:
            output_file.write(f"{original_image_name},{image_id}\n")

    print(f"Sorted unique data saved successfully to {output_path}")


if __name__ == '__main__':
    csv_file_path = 'output.csv'  # CSV 文件路径

    # 调用函数来保存数据
    save_sorted_unique_data_to_txt(csv_file_path)