
from Layers.myllamalayers import KVCache_Attention, MultiHeadAttention,ResidueBlock

from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, Multiply, Concatenate, Add, Conv2D, Conv2DTranspose, LayerNormalization, Dropout
from tensorflow.keras import Model
from keras_nlp.layers import TokenAndPositionEmbedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CosineSimilarity, MeanSquaredError, SparseCategoricalCrossentropy




class MyLLama(Model):

    def __init__(self, embedDim, numHeads):

        super(MyLLama, self).__init__()

        self.embedDim = embedDim
        self.numHeads = numHeads

        self.embedLayer = TokenAndPositionEmbedding(100, 10, self.embedDim)
        self.residualBlocks = [ResidueBlock(self.embedDim, self.numHeads) for i in range(20)]
        self.denseLayers = [Dense(10, activation="silu") for i in range(7)]
        self.selfAttention=KVCache_Attention(self.embedDim)

        self.loss = CosineSimilarity()
        self.optimizer = Adam()

    def loss_function(self, logits, target):

        loss = tf.keras.losses.sparse_categorical_crossentropy(target, logits, from_logits=True)

        return loss

    def call(self, x, target=None, reset=False):

        x = tf.keras.utils.pad_sequences(x, 10)[0]

        x = self.embedLayer(x)

        for layer in self.residualBlocks:
            x = layer(x, tf.Variable(reset))

        x=self.selfAttention(x)

        for layer in self.denseLayers:
            x = layer(x)

        logits = tf.nn.softmax(x[:, -1])

        if target is not None:

            loss = self.loss_function(x, target)

        else:

            loss = None

        return logits, loss

    def train(self,x,y):

        x = # Data Base Connectors
        y = # Data Base Connectors

        for i in range(10):
            with tf.GradientTape() as tape:
                logits, loss = self.call(x, y)

                print(f"Loss at step {i}: ", float(tf.reduce_mean(loss)))

                grads = tape.gradient(loss, self.trainable_variables)

            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # reset with start token
        self.call(x=tf.constant([[0]]), reset=tf.Variable(True))

        return "Training Comleted"