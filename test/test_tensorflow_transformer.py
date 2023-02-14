import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.data import Dataset


class Transformer(tf.keras.Model):
    def __init__(self, d_model=128, nhead=8, num_layers=6, dim_feedforward=512):
        super(Transformer, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder = tf.keras.layers.TransformerEncoder(
            num_layers, d_model, nhead, dim_feedforward)
        self.fc = Dense(768*768*1280, activation='sigmoid')

    def call(self, inputs):
        x = self.pos_encoder(inputs)
        output = self.transformer_encoder(x)
        output = Flatten()(output)
        output = self.fc(output)
        output = Reshape((768, 768, 1280))(output)
        return output


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = tf.zeros((max_len, d_model))
        position = tf.range(0, max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32)
                          * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = tf.sin(position * div_term)
        pe[:, 1::2] = tf.cos(position * div_term)
        pe = pe[tf.newaxis, ...]
        self.pe = tf.Variable(pe, trainable=False)

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]


# Create a random volume for demonstration purposes
volume = tf.random.normal((1, 768, 768, 1280))

# Create a Transformer model and pass the volume through it
transformer = Transformer()
output = transformer(volume)

# Print the shape of the output
print(output.shape)
