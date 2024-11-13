from keras.models import model_from_json
import os
from IPython.display import Image
import config
import scipy
from utils import grouper,get_questions_tensor_timeseries,get_images_matrix,get_answers_matrix
import numpy as np
# 在新的环境下：

# 载入NLP的模型
nlp = config.nlp
# 以及label的encoder
labelencoder = config.labelencoder

# 接着，把模型读进去
model = model_from_json(open(config.open_model).read())
model.load_weights(config.oprn_weight)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


flag = True

# 所有需要外部导入的料
caffe = '/home/ubuntu/Downloads/caffe'
vggmodel = r'D:\Ai_zhiyi\datasets\preprocessed\date\VGG_ILSVRC_19_layers.caffemodel'
prototxt = r'D:\Ai_zhiyi\datasets\preprocessed\date\VGG.prototxt'
img_path = r'D:\Ai_zhiyi\datasets\preprocessed\date\test_img.png'
image_features = r'D:\Ai_zhiyi\datasets\preprocessed\date\test_img_vgg_feats.mat'

while flag:
    # 首先，给出你要提问的图片
    img_path = str(input('Enter path to image : '))
    # 对于这个图片，我们用caffe跑一遍VGG CNN，并得到4096维的图片特征
    os.system('python extract_features.py --caffe ' + caffe + ' --model_def ' + prototxt + ' --model ' + vggmodel + ' --image ' + img_path + ' --features_save_to ' + image_features)
    print ('Loading VGGfeats')
    # 把这个图片特征读入
    features_struct = scipy.io.loadmat(image_features)
    VGGfeatures = features_struct['feats']
    print ("Loaded")
    # 然后，你开始问他问题
    question = input("Ask a question: ")
    if question == "quit":
        flag = False
    timesteps = config.max_len
    X_q = get_questions_tensor_timeseries([question], nlp, timesteps)
    X_i = np.reshape(VGGfeatures, (1, 4096))
    # 构造成input形状
    X = [X_q, X_i]
    # 给出prediction
    # 获取模型的预测概率
    y_pred_prob = model.predict(X, verbose=0)

    # 将概率转换为类别
    y_predict = np.argmax(y_pred_prob, axis=1)
    print (labelencoder.inverse_transform(y_predict))