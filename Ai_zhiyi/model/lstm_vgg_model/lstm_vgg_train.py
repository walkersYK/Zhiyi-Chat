import scipy.io
from keras.src.utils import np_utils, generic_utils
from datasets import NewData
from utils import grouper,get_questions_tensor_timeseries,get_images_matrix,get_answers_matrix
from model.lstm_vgg_model.lstm import model,model_file_name
import config
import numpy as np

max_len = 30
word_vec_dim = 300
img_dim = 4096
dropout = 0.5
activation_mlp = 'tanh'
num_epochs = 1
model_save_interval = 5

num_hidden_units_mlp = 1024
num_hidden_units_lstm = 512
num_hidden_layers_mlp = 3
num_hidden_layers_lstm = 1
batch_size = 128

nb_classes = 557  # 假设类别数量
questions_train,answers_train,images_train = NewData(config.questions_train,config.answers_train,config.images_train).get_data()

features_struct = np.load(config.npz_file_path)
VGGfeatures = features_struct['features']
VGGfeatures = VGGfeatures.T

# To Do 这里的VGGfeatures =》 【img_num, 4096】


print('loaded vgg features')

image_ids = open(config.vgg_img_id_map).read().splitlines()
img_map = {}
# print(len(image_ids))
for ids in image_ids:
    id_split = ids.split()
    img_map[id_split[0]] = int(id_split[1])
nlp = config.nlp
print('loaded word2vec features...')

## training
print('Training started...')

for k in range(config.num_epochs):

    progbar = generic_utils.Progbar(len(questions_train))

    for qu_batch, an_batch, im_batch in zip(grouper(questions_train, batch_size, fillvalue=questions_train[-1]),
                                            grouper(answers_train, batch_size, fillvalue=answers_train[-1]),
                                            grouper(images_train, batch_size, fillvalue=images_train[-1])):
        X_q_batch = get_questions_tensor_timeseries(qu_batch, nlp, max_len)
        X_i_batch = get_images_matrix(im_batch, img_map, VGGfeatures)
        Y_batch = get_answers_matrix(an_batch, config.labelencoder)
        print(f"X_q_batch shape: {X_q_batch.shape}")
        print(f"X_i_batch shape: {X_i_batch.shape}")
        print(f"Y_batch shape: {Y_batch.shape}")
        loss = model.train_on_batch([X_q_batch, X_i_batch], Y_batch)
        progbar.add(batch_size, values=[("train loss", loss)])

    if k % model_save_interval == 0:
        model.save_weights(model_file_name + '_epoch_{:03d}.hdf5'.format(k))

model.save_weights(model_file_name + '_epoch_{:03d}.hdf5'.format(k))