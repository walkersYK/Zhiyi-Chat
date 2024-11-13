import spacy
import joblib
"""
原始语料
"""
questions_train = open(r'D:\Ai_zhiyi\datasets\preprocessed\questions_train.txt', 'r').read().splitlines()
answers_train = open(r'D:\Ai_zhiyi\datasets\preprocessed\answers_train.txt', 'r').read().splitlines()
images_train = open(r'D:\Ai_zhiyi\datasets\preprocessed\output.txt', 'r').read().splitlines()

vgg_model_path = r'D:\Ai_zhiyi\datasets\vgg_feats.npz'

# 载入Spacy的英语库
nlp = spacy.load("en_core_web_md")

"""模型训练"""
num_epochs = 1
max_len = 30

labelencoder = joblib.load(r'D:\Ai_zhiyi\datasets\preprocessed\date\labelencoder.pkl')

### 模型文件###
npz_file_path = r"D:\Ai_zhiyi\datasets\vgg_feats.npz"
vgg_img_id_map = r"D:\Ai_zhiyi\datasets\preprocessed\img_vgg_IDMap.txt"

###读取文件###
open_model = r"D:\Ai_zhiyi\model\lstm_vgg_model\lstm_1_num_hidden_units_lstm_512_num_hidden_units_mlp_1024_num_hidden_layers_mlp_3_num_hidden_layers_lstm_1.json"
oprn_weight = r"D:\Ai_zhiyi\model\lstm_vgg_model\lstm_1_num_hidden_units_lstm_512_num_hidden_units_mlp_1024_num_hidden_layers_mlp_3_num_hidden_layers_lstm_1_epoch_000.hdf5"