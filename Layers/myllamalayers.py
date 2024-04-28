import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, Multiply, Concatenate, Add, Conv2D, Conv2DTranspose, LayerNormalization, Dropout
from tensorflow.keras import Model
from keras_nlp.layers import TokenAndPositionEmbedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CosineSimilarity, MeanSquaredError, SparseCategoricalCrossentropy


class KVCache_Attention(Layer):

    def __init__(self, inner_dim):

        super(KVCache_Attention, self).__init__()

        self.KCache = None
        self.VCache = None
        self.inner_dim = inner_dim

        self.key_projection = Dense(self.inner_dim, activation="linear")
        self.value_projection = Dense(self.inner_dim, activation="linear")
        self.query_projection = Dense(self.inner_dim, activation="linear")
        self.dropout = Dropout(0.1)
        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()
        self.add = Add()

        self.out_projection_1 = Dense(self.inner_dim, activation="silu")
        self.out_projection_2 = Dense(self.inner_dim, activation="linear")

    def call(self, x, reset=False):

        x = self.ln1(x)

        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        if self.KCache is None or reset:

            self.KCache = tf.Variable(key, trainable=False)
            self.VCache = tf.Variable(value, trainable=False)

        else:

            self.KCache = tf.concat([self.KCache, key], axis=0)
            self.VCache = tf.concat([self.VCache, value], axis=0)

        keys = tf.matmul(query, self.KCache, transpose_b=True)
        residue = tf.matmul(keys, self.VCache)

        value = self.out_projection_1(residue)
        value = self.ln2(value)
        value = tf.nn.softmax(value)

        value = self.out_projection_2(value)

        return value


class MultiHeadAttention(Layer):

    def __init__(self, embedDim, numHeads):
        super(MultiHeadAttention, self).__init__()

        self.embedDim = embedDim
        self.numHeads = numHeads
        self.headDim = embedDim // numHeads

    def build(self):
        self.attention_blocks = [KVCache_Attention(self.headDim) for i in range(self.numHeads)]

    def call(self, x, reset=False):
        i = 0
        c = []

        for layer in self.attention_blocks:
            v = layer(x[:, i:i + self.headDim], reset)
            c.append(v)
            i += self.headDim

        c = tf.concat(c, axis=-1)

        return c


class ResidueBlock(Layer):

    def __init__(self, embedDim, numHeads):
        super(ResidueBlock, self).__init__()
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.mhaBlock = MultiHeadAttention(self.embedDim, self.numHeads)

    def call(self, x, reset=False):
        residue = x

        x = self.mhaBlock(x, reset)

        return tf.nn.swish(x + residue)



