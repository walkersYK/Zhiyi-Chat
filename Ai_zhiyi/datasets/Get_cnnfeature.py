import tensorflow as tf
import os
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
# 定义模型
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
model = Model(inputs=base_model.input, outputs=x)

image_folder = r'D:\Ai_zhiyi\VQA_Red_data\VQA_RAD Image Folder'
output_file = 'vgg_feats.npz'

# 检查 .npz 文件是否存在
if not os.path.exists(output_file):
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 初始化特征矩阵
    num_images = len(image_files)
    features_matrix = np.zeros((num_images, 4096), dtype=np.float32)

    # 批量处理图像
    for i, img_path in enumerate(image_files):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # 提取特征
        features = model.predict(img_array)
        features_matrix[i] = features.flatten()

    np.savez(output_file, features=features_matrix, filenames=image_files)

    print("特征提取完成，已保存到 vgg_feats.npz")

if __name__ == '__main__':
    npz_file_path = 'vgg_feats.npz'
    # 加载 .npz 文件
    data = np.load(npz_file_path)

    features = data['features']
    print("Features shape:", features[:2])