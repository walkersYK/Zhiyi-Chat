import csv
import os


def save_data_from_csv(csv_file_path, directory='extracted_data'):
    # 创建目录如果它不存在
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 输出文本文件路径
    image_ids_output_path = os.path.join(directory, 'image_ids_train.txt')
    questions_output_path = os.path.join(directory, 'questions_train.txt')
    answers_output_path = os.path.join(directory, 'answers_train.txt')
    img_name_output_path = os.path.join(directory, 'img_name.txt')

    # 初始化文件（清空旧内容）
    open(image_ids_output_path, 'w').close()
    open(questions_output_path, 'w').close()
    open(answers_output_path, 'w').close()
    open(img_name_output_path, 'w').close()

    # 打开 CSV 文件并读取数据
    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        # 创建三个空列表来存储 image_id、问题和答案
        image_ids = []
        questions = []
        answers = []
        img_names = []

        # 遍历每一行数据
        for row in reader:
            image_ids.append(row['image_id'])
            questions.append(row['question'])
            answers.append(row['answer'])
            img_names.append(row['original_image_name'])

    # 将 image_id 写入文本文件
    with open(image_ids_output_path, mode='w', encoding='utf-8') as image_ids_file:
        for image_id in image_ids:
            image_ids_file.write(f"{image_id}\n")

    # 将问题写入文本文件
    with open(questions_output_path, mode='w', encoding='utf-8') as questions_file:
        for question in questions:
            questions_file.write(f"{question}\n")

    # 将答案写入文本文件
    with open(answers_output_path, mode='w', encoding='utf-8') as answers_file:
        for answer in answers:
            answers_file.write(f"{answer}\n")

    with open(img_name_output_path, mode='w', encoding='utf-8') as img_name_file:
        for img_name in img_names:
            img_name_file.write(f"{img_name}\n")

    print(f"Image IDs saved successfully to {image_ids_output_path}")
    print(f"Questions saved successfully to {questions_output_path}")
    print(f"Answers saved successfully to {answers_output_path}")
    print(f"Answers saved successfully to {img_name_output_path}")


if __name__ == '__main__':
    csv_file_path = 'output.csv'  # CSV 文件路径

    # 调用函数来保存数据
    save_data_from_csv(csv_file_path)