import tensorflow as tf
from tensorflow.keras.models import Model

import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPool1D, LeakyReLU, ReLU, Dropout, BatchNormalization, GroupNormalization


from tensorflow.python.keras import Sequential

def make_naive_discriminator(seq_shape:tuple, n_classes:int)-> Model:

    _input = Input(seq_shape)

    x = Conv1D(8, 5, 2, padding='same')(_input)
    x = Dropout(0.25)(x) 
    x = LeakyReLU()(x)

    x = Conv1D(16, 5, 2, padding='same')(x)
    x = Dropout(0.25)(x) 
    x = LeakyReLU()(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = Dense(32)(x)
    x = LeakyReLU()(x)
    
    x = Dense(n_classes, activation="softmax")(x)

    model = Model(_input, x)
    
    model.compile("adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
    
    return model