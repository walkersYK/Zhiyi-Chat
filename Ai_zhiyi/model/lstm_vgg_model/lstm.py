import tensorflow as tf
from IPython.display import Image
from tensorflow.python.keras.layers import  Concatenate

"""
Sequential 模型只能依次添加层，而不能直接处理多个输入或中间层的拼接。
"""


# 参数定义
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

# 图片模型，也就是专门用来处理图片部分
image_input = tf.keras.layers.Input(shape=(img_dim,))
image_model = tf.keras.layers.Reshape((img_dim,))(image_input)

# 语言模型，专门用来处理语言
language_input = tf.keras.layers.Input(shape=(max_len, word_vec_dim))
if num_hidden_layers_lstm == 1:
    language_model = tf.keras.layers.LSTM(units=num_hidden_units_lstm, return_sequences=False,
                                          input_shape=(max_len, word_vec_dim))(language_input)
else:
    language_model = tf.keras.layers.LSTM(units=num_hidden_units_lstm, return_sequences=True,
                                          input_shape=(max_len, word_vec_dim))(language_input)
    for i in range(num_hidden_layers_lstm - 2):
        language_model = tf.keras.layers.LSTM(units=num_hidden_units_lstm, return_sequences=True)(language_input)
    language_model = tf.keras.layers.LSTM(units=num_hidden_units_lstm, return_sequences=False)(language_input)


# 合并两个模型的输出
merged_output = Concatenate(axis=1)([language_model, image_model])

# 添加 MLP 层
for i in range(num_hidden_layers_mlp):
    merged_output = tf.keras.layers.Dense(num_hidden_units_mlp, kernel_initializer='uniform')(merged_output)
    merged_output = tf.keras.layers.Activation(activation_mlp)(merged_output)
    merged_output = tf.keras.layers.Dropout(dropout)(merged_output)

# 添加最终的分类层
merged_output = tf.keras.layers.Dense(nb_classes)(merged_output)
merged_output = tf.keras.layers.Activation('softmax')(merged_output)

# 创建最终模型
model = tf.keras.models.Model(inputs=[language_input, image_input], outputs=merged_output)

# 保存模型结构
json_string = model.to_json()
model_file_name = 'lstm_1_num_hidden_units_lstm_' + str(num_hidden_units_lstm) + \
                    '_num_hidden_units_mlp_' + str(num_hidden_units_mlp) + '_num_hidden_layers_mlp_' + \
                    str(num_hidden_layers_mlp) + '_num_hidden_layers_lstm_' + str(num_hidden_layers_lstm)
with open(model_file_name + '.json', 'w') as json_file:
    json_file.write(json_string)

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print('Model compiled.')

# 绘制模型图
tf.keras.utils.plot_model(model, to_file='model_lstm.png', show_shapes=True)
Image(filename='model_lstm.png')