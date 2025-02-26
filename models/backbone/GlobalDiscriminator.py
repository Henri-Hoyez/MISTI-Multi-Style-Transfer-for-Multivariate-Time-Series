import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf

# from tensorflow.python.keras import Input
# from tensorflow.python.keras.layers import Conv1D, LeakyReLU, Dense, Dropout, Flatten, Concatenate
# from tensorflow.python.keras.models import Model

from tensorflow.keras import Input
from tensorflow.keras.layers import SpectralNormalization
from tensorflow.keras.layers import Conv1D, LeakyReLU, Dense, Dropout, Flatten, Concatenate
from tensorflow.keras.models import Model


def discr_downsampling(x, n_filters, dropout=0.0):
    
    initializer = tf.keras.initializers.GlorotNormal(seed=42)
    
    x = SpectralNormalization(Conv1D(n_filters, 5, 2, padding='same', kernel_initializer=initializer))(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)
    return x

def make_global_discriminator(seq_length:int, n_signals:int, n_classes:int, filters:list, dropout: int=0.0):
    initializer = tf.keras.initializers.RandomNormal(seed=42)
    
    inputs = [Input((seq_length, 1)) for _ in range(n_signals)]

    _input = Concatenate(-1)(inputs)

    x = discr_downsampling(_input, filters[0], dropout=dropout)

    for f in filters[1:]:
        x = discr_downsampling(x, f, dropout=dropout)
        
        
    _output = SpectralNormalization(Conv1D(1, 3, 1, padding='same', kernel_initializer=initializer))(x)

    flatened = Flatten()(x)
    flatened = Dropout(dropout)(flatened)
    
    # crit_hidden_layer = Dense(10, kernel_initializer=initializer)(flatened)
    # _output = Dense(1, activation="linear", kernel_initializer=initializer)(crit_hidden_layer)

    class_hidden = Dense(150, kernel_initializer=initializer)(flatened)
    class_hidden = LeakyReLU()(class_hidden)
    
    class_hidden = Dense(50, kernel_initializer=initializer)(class_hidden)
    class_hidden = LeakyReLU()(class_hidden)
    
    _class_output = Dense(n_classes, activation="softmax", kernel_initializer=initializer)(class_hidden)

    model = Model(inputs, [_output, _class_output], name="global_discriminator")
    
    

    return model


def make_univariate_global_discriminator(seq_length:int, n_signals:int, n_classes:int):
    inputs = [Input((seq_length, 1)) for _ in range(n_signals)]

    _input = Concatenate(-1)(inputs)

    x = discr_downsampling(_input, 8)

    x = discr_downsampling(x, 16)

    flatened = Flatten()(x)
    flatened = Dropout(0.0)(flatened)
    
    crit_hidden_layer = Dense(10)(flatened)
    _output = Dense(1, activation="linear")(crit_hidden_layer)
    
    class_hidden = Dense(50)(flatened)
    class_hidden = LeakyReLU()(class_hidden)

    _class_output = Dense(n_classes, activation="softmax")(class_hidden)

    model = Model(inputs, [_output, _class_output], name="global_discriminator")

    return model

