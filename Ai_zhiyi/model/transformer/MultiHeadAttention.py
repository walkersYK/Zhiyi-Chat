import tensorflow as tf
def split_heads(self, x, batch_size):
    """分拆最后一个维度到 (num_heads, depth).
    转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
# ...
# # 这里记得要把 batch_size 传进去
# q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
# k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
# v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
# ...
def scaled_dot_product_attention(q, k, v, mask):
    """ 这部分是对于输入的张量维度进行描述

    1. q, k, v 必须具有匹配的前置维度。

    2. k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。

    3. 虽然 mask 根据其类型（填充或前瞻）有不同的形状，但是 mask 必须能进行广播转换以便求和。


    参数:
      q: 请求的形状 == (..., seq_len_q, depth)
      k: 主键的形状 == (..., seq_len_k, depth)
      v: 数值的形状 == (..., seq_len_v, depth_v)
      mask: Float 张量，其形状能转换成
            (..., seq_len_q, seq_len_k)。默认为None。

    返回值:
      输出，注意力权重
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # 得到张量形状为 (..., seq_len_q, seq_len_k)

    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)  # dk 即 depth
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
        # 相加等于1。
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        """
          参数: d_model 必须能被 num_heads 整除
          d_model: 由于要映射 num_heads 组 Q,K,V. d_model 的值需要为 num_heads * depth
          num_heads: 代表注意力机制的头数
        """

        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)


def split_heads(self, x, batch_size):
    """分拆最后一个维度到 (num_heads, depth).
    转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])


def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    # 这里采取将 q,k,v 先线性变换到 (..., seq_len, d_model)
    # 再拆分成 num_heads 份 (..., nums_heads, seq_ken, d_model / nums_heads)
    # 这和直接将原始 q,k,v 线性变换成 nums_heads 组是等效的！这样写效率更高！

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights
