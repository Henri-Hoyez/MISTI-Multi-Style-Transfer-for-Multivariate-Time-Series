import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ["TF_USE_LEGACY_KERAS"]="1"
# import tensorflow as tf

# from tensorflow.python.keras import Input
# from tensorflow.python.keras.layers import Layer
# from tensorflow.python.keras.initializers import RandomNormal
# from tensorflow.python.keras.layers import Conv1D, LeakyReLU, Dense, Conv1DTranspose, Reshape, Flatten
# from tensorflow.python.keras.models import Model

from tensorflow.keras.layers import BatchNormalization, SpectralNormalization # type: ignore
from tensorflow.keras import Input # type: ignore 
from tensorflow.keras.layers import SpectralNormalization # type: ignore
from tensorflow.keras.layers import Conv1D, LeakyReLU, Dense, Dropout, Flatten, Concatenate, Conv1DTranspose, Reshape, Add# type: ignore
from tensorflow.keras.models import Model# type: ignore
from tensorflow.keras.regularizers import L2 # type: ignore
from tensorflow.keras.layers import Layer # type: ignore
from tensorflow.keras.initializers import RandomNormal 

from tensorflow.keras.layers import Add

from models.Layers.AdaIN import AdaIN


def linear_projection(style_input:Layer, actual_seq_len:int):

    adapter = Dense(actual_seq_len)(style_input)
    adapter = Reshape((actual_seq_len, 1))(adapter)

    return adapter

def upsampling_block(content_input:Layer, style_input:Layer, filters):
    
    # Upsample
    x_first = Conv1DTranspose(filters, 5, 2, padding='same')(content_input)
    actual_sequence_len = x_first.shape[1]

    adapted_style_input = linear_projection(style_input, actual_sequence_len) # Stage 1
    x = AdaIN()(x_first, adapted_style_input)
    x = LeakyReLU()(x)

    x = Conv1DTranspose(filters, 3, 1, padding='same')(x)
    adapted_style_input = linear_projection(style_input, actual_sequence_len)  # Stage 2
    x = AdaIN()(x, adapted_style_input)
    x = LeakyReLU()(x)
    
    x = Conv1DTranspose(filters, 3, 1, padding='same')(x)
    adapted_style_input = linear_projection(style_input, actual_sequence_len) # Stage 3
    x = AdaIN()(x, adapted_style_input)
    x = LeakyReLU()(x)
    
    x = Concatenate()([x, x_first])
    
    return x


def generator_part(content_inputs, style_input, filters:list):
    
    x = upsampling_block(content_inputs, style_input, filters[0]) # 16
    
    for f in filters[1:]:
        x = upsampling_block(x, style_input, f) # 32
    
    # output
    x = Conv1DTranspose(1, 3, 1, padding='same')(x)
    
    return x

def make_generator(n_sample_wiener:int, feat_wiener:int, style_vector_size:int, n_generators:int, filters: list):
    init = RandomNormal()

    content_input = Input((n_sample_wiener, feat_wiener))
    style_input = Input((style_vector_size,), name="Style_Input") 
    gens_outputs = []


    for _ in range(n_generators):
        gens_outputs.append(generator_part(content_input, style_input, filters))
        # break

    model = Model([content_input, style_input], gens_outputs)
        
    return model

# univariate Decoder.

def univariate_generator_part(content_inputs, style_input):

    x = upsampling_block(content_inputs, style_input, 16) # 16

    x = upsampling_block(x, style_input, 16) # 32
    
    # x = upsampling_block(x, style_input, 16) # 32
    
    # output
    x = Conv1DTranspose(1, 3, 1, padding='same')(x)
    
    return x

def make_univariate_generator(n_sample_wiener:int, feat_wiener:int, style_vector_size:int, n_generators:int):
    init = RandomNormal()

    content_input = Input((n_sample_wiener, feat_wiener))
    style_input = Input((style_vector_size,), name="Style_Input") 
    gens_outputs = []


    for _ in range(n_generators):
        gens_outputs.append(univariate_generator_part(content_input, style_input))
        # break

    model = Model([content_input, style_input], gens_outputs)
        
    return model

