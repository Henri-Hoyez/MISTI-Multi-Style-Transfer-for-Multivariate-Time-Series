import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf
import keras

# from tensorflow.python.keras.regularizers import l2
# from tensorflow.python.keras import Input
# from tensorflow.python.keras.layers import Conv1D, LeakyReLU, Dense, Dropout, Flatten
# from tensorflow.python.keras.models import Model

from tensorflow.keras.layers import BatchNormalization, SpectralNormalization # type: ignore
from tensorflow.keras import Input # type: ignore 
from tensorflow.keras.layers import SpectralNormalization # type: ignore
from tensorflow.keras.layers import Conv1D, LeakyReLU, Dense, Dropout, Flatten, Concatenate# type: ignore
from tensorflow.keras.models import Model# type: ignore


def discr_downsampling(x, n_filters, dropout:int=0.):
    initializer = tf.keras.initializers.GlorotNormal(seed=42)

    x = SpectralNormalization(Conv1D(n_filters, 5, 2, padding='same', kernel_initializer=initializer))(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)
    return x


def local_discriminator_part(_input, filters: list, dropout:int=0.):    

    x = discr_downsampling(_input, filters[0], dropout)
    
    for f in filters[1:]:
        x = discr_downsampling(x, f)
    

    x = Flatten()(x)
    stage1_dropouted = Dropout(dropout)(x)

    _output = Dense(1, activation="linear")(stage1_dropouted)

    return _output

def create_local_discriminator(n_signals:int, sequence_length:int, filters: list, dropout:int=0.):

    inputs = [Input((sequence_length, 1)) for _ in range(n_signals)]
    crit_outputs = []

    for sig_input in inputs:
        crit_output = local_discriminator_part(sig_input, filters, dropout)
        crit_outputs.append(crit_output)

    return Model(inputs, crit_outputs)

##
# UNIVARIATE Ds #
##

def univariate_local_discriminator_part(_input):
    initializer = tf.keras.initializers.GlorotNormal(seed=42)

    x = discr_downsampling(_input, 8)
    
    x = discr_downsampling(x, 16)

    # x = Flatten()(x)
    # stage1_dropouted = Dropout(0.3)(x)

    # _output = Dense(1, activation="linear")(stage1_dropouted)
    
    _output = SpectralNormalization(Conv1D(1, 3, 1, padding='same', kernel_initializer=initializer))(x)
    
    return _output

def create_univariate_local_discriminator(n_signals:int, sequence_length:int):

    inputs = [Input((sequence_length, 1)) for _ in range(n_signals)]
    crit_outputs = []

    for sig_input in inputs:
        crit_output = univariate_local_discriminator_part(sig_input)
        crit_outputs.append(crit_output)

    return Model(inputs, crit_outputs)