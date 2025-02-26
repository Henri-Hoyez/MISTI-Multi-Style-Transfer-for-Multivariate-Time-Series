import tensorflow as tf
from tensorflow.keras.models import Model

import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, MaxPool1D, ReLU, Dropout, BatchNormalization,  Conv1DTranspose 
from tensorflow.python.keras import Sequential


def compress(input_tensor:tf.keras.layers.Layer, filters):
    x = Conv1D(filters, 5, 2, padding='same')(input_tensor)
    # x = Dropout(0.2)(x) 
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

def uncompress(input_tensor:tf.keras.layers.Layer, filters):
    x = Conv1DTranspose(filters, 5, 2, padding='same')(input_tensor)
    # x = Dropout(0.2)(x) 
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def make_naive_ae(seq_shape:tuple)-> Model:

    _input = Input(seq_shape)

    x = compress(_input, 8)
    x = compress(x, 16)
    x = compress(x, 32)

    x = uncompress(x, 32)
    x = uncompress(x, 16)
    x = uncompress(x, seq_shape[-1])

    model = Model(_input, x)
    
    model.compile("adam", loss="mae", metrics=["mae"])
        
    return model

