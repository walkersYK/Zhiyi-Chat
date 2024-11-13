from keras.src.utils import np_utils, generic_utils
import scipy.io
import numpy as np
import spacy
from itertools import zip_longest


# 读取特征矩阵
features_matrix = np.load(r'D:\Ai_zhiyi\datasets\vgg_feats.npz')
vgg_model_path = '../Downloads/coco/vgg_feats.mat'
# 导入下载好的vgg_fearures (这是个matlab的文档，没关系，scipy可以读取)
features_matrix = features_matrix['features']
# print(features_matrix[:2])



# 把所有的answers转化成数字化的label
def get_answers_matrix(answers, encoder):
    # string转化成数字化表达
    print(answers)
    print(encoder)
    y = encoder.transform(answers)
    nb_classes = encoder.classes_.shape[0]
    Y = np_utils.to_categorical(y, nb_classes)
    # 并构造成标准的matrix
    return Y


def img_vgg():
    # 读入VGG features
    features_struct = scipy.io.loadmat(vgg_model_path)
    VGGfeatures = features_struct['feats']
    # 跟图片一一对应
    image_ids = open('../data/coco_vgg_IDMap.txt').read().splitlines()
    id_map = {}
    for ids in image_ids:
        id_split = ids.split()
        id_map[id_split[0]] = int(id_split[1])

# 取得任何一个input图片的“数字化表达形式”
def get_images_matrix(img_coco_ids, img_map, VGGfeatures):
    nb_samples = len(img_coco_ids)
    nb_dimensions = VGGfeatures.shape[0]
    image_matrix = np.zeros((nb_samples, nb_dimensions))
    for j in range(len(img_coco_ids)):
        image_matrix[j,:] = VGGfeatures[:,img_map[img_coco_ids[j]]]
    return image_matrix

# 问句中的所有英文转化为vector，并平均化整个句子

# 图片的维度大小
img_dim = 4096
# 句子/单词的维度大小
word_vec_dim = 300

# 这个method就是用来计算句子中所有word vector的总和，
# 目的在于把我们的文字用数字表示
def get_questions_matrix_sum(questions, nlp):
    # assert not isinstance(questions, basestring)
    nb_samples = len(questions)
    word_vec_dim = nlp(questions[0])[0].vector.shape[0]
    questions_matrix = np.zeros((nb_samples, word_vec_dim))
    for i in range(len(questions)):
        tokens = nlp(questions[i])
        for j in range(len(tokens)):
            questions_matrix[i,:] += tokens[j].vector
    return questions_matrix

def get_questions_tensor_timeseries(questions, nlp, timesteps):
    nb_samples = len(questions)
    word_vec_dim = nlp(questions[0])[0].vector.shape[0]
    questions_tensor = np.zeros((nb_samples, timesteps, word_vec_dim))
    for i in range(len(questions)):
        tokens = nlp(questions[i])
        for j in range(len(tokens)):
            if j<timesteps:
                questions_tensor[i,j,:] = tokens[j].vector

    return questions_tensor

def get_img_map():
    image_ids = open(r'D:\Ai_zhiyi\datasets\preprocessed\img_vgg_IDMap.txt').read().splitlines()
    img_map = {}
    for ids in image_ids:
        id_split = ids.split()
        img_map[id_split[0]] = int(id_split[1])
# 这是一个标准的chunk list方法
#  "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)