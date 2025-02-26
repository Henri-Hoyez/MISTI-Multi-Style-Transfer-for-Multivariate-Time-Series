import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"

import tensorflow as tf
# from tensorflow.python.keras import Input
# from tensorflow.python.keras.layers import Conv1D, LeakyReLU, Flatten, Dense
# from keras.src.layers.normalization.group_normalization import GroupNormalization

# from tensorflow.python.keras.models import Model

from tensorflow.keras.layers import SpectralNormalization, GroupNormalization # type: ignore
from tensorflow.keras import Input # type: ignore 
from tensorflow.keras.layers import Conv1D, LeakyReLU, Dense, Dropout, Flatten, Concatenate# type: ignore
from tensorflow.keras.models import Model# type: ignore
from tensorflow.keras import Layer # type: ignore

class NormLayer(Layer):
    def call(self, x):
        return tf.clip_by_norm(x, 1., -1)


def style_downsampling_block(x:Layer, n_filters:int):
    x = tf.keras.layers.Conv1D(n_filters, 5, 2, padding='same')(x) 
    x = tf.keras.layers.GroupNormalization(groups=-1)(x)
    return tf.keras.layers.LeakyReLU()(x)


def make_style_encoder(seq_length:int, n_feat:int, vector_output_shape:int, filters: list)  -> Model:

    _input = tf.keras.Input((seq_length, n_feat))

    x = style_downsampling_block(_input, filters[0])
    
    # ###
    for filter in filters[1:]:
        x = style_downsampling_block(x, filter)

    # ###
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(vector_output_shape)(x)
    
    x = NormLayer()(x)

    model = tf.keras.Model(_input, x)

    return model


def make_univariate_style_encoder(seq_length:int, n_feat:int, vector_output_shape:int)  -> Model:

    _input = tf.keras.Input((seq_length, n_feat))

    x = style_downsampling_block(_input, 16)
    
# ###
    x = style_downsampling_block(x, 32)

# ###
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(vector_output_shape)(x)
    
    x = NormLayer()(x)

    model = tf.keras.Model(_input, x)
    
    return model