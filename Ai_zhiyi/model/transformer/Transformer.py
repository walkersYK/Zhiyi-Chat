import tensorflow as tf
from .encoder import Encoder
from .Decoder import Decoder
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def _call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        """

        :param inp:输入序列，这里需要的是源语言（葡萄牙语）的编码表示。（嵌入表示将在编码器中完成）
        :param tar:目标序列，这里需要的是目标语言（英语）的编码表示。（嵌入表示将在编码器中完成）
        :param training:目标序列，这里需要的是目标语言（英语）的编码表示。（嵌入表示将在编码器中完成）
        :param enc_padding_mask:编码器，填充遮挡。
        :param look_ahead_mask:前瞻遮挡。两个遮挡将在后面详细描述。
        :param dec_padding_mask:解码器，填充遮挡。
        :return:
        """
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
