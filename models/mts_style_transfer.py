import tensorflow as tf
from utils.Layers.AdaIN import AdaIN


## 
# COSCI Like Generator
## 

def generator_part(content_input, n_sample_wiener:int, feat_wiener:int, style_input):
    init = tf.keras.initializers.RandomNormal()

    # Make a small projection...
    _content_input = tf.keras.layers.Flatten()(content_input)
    _content_input = tf.keras.layers.Dense(n_sample_wiener* feat_wiener, kernel_initializer=init)(_content_input)
    _content_input = tf.keras.layers.Reshape((n_sample_wiener, feat_wiener))(_content_input)

    # Make the style input 
    _style_input = tf.keras.layers.Dense(16)(style_input)
    _style_input = tf.keras.layers.Reshape((16, 1))(_style_input)

    _stage2_style_input = tf.keras.layers.Dense(32)(style_input)
    _stage2_style_input = tf.keras.layers.Reshape((32, 1))(_stage2_style_input)


    x = tf.keras.layers.Conv1DTranspose(64, 5, 1, padding='same', kernel_initializer=init)(_content_input)
    x = AdaIN()(x, _style_input)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1DTranspose(64, 5, 1, padding='same', kernel_initializer=init)(x)
    x = AdaIN()(x, _style_input)
    x = tf.keras.layers.LeakyReLU()(x)  

    x = tf.keras.layers.Conv1DTranspose(64, 5, 2, padding='same', kernel_initializer=init)(x)
    x = tf.keras.layers.LeakyReLU()(x)


    x = tf.keras.layers.Conv1DTranspose(32, 5, 1, padding='same', kernel_initializer=init)(x)
    x = AdaIN()(x, _stage2_style_input)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1DTranspose(32, 5, 1, padding='same', kernel_initializer=init)(x)
    x = AdaIN()(x, _stage2_style_input)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1DTranspose(32, 5, 2, padding='same', kernel_initializer=init)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # output
    x = tf.keras.layers.Conv1DTranspose(1, 5, 1, padding='same', kernel_initializer=init)(x)
    # x = tf.keras.layers.LeakyReLU()(x)

    return x

def make_generator(n_sample_wiener:int, feat_wiener:int, style_vector_size:int, n_generators:int):
    
    input = tf.keras.Input((n_sample_wiener, feat_wiener), name=f"Content_Input")
    style_input = tf.keras.Input((style_vector_size,), name="Style_Input") 
    gens_outputs = []

    for _ in range(n_generators):
        gens_outputs.append(generator_part(input, n_sample_wiener, feat_wiener, style_input))

    test = tf.keras.layers.concatenate(gens_outputs, axis=-1)

    model = tf.keras.Model([input, style_input], test)

    return model



##
# COSCI-Like Discriminator
##

def make_global_discriminator(seq_length:int, n_signals:int, n_classes:int):
    _input = tf.keras.Input((seq_length, n_signals))

    x = tf.keras.layers.Conv1D(32, 5, 2, padding='same')(_input)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1D(32, 5, 2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Flatten()(x)
    crit_hidden_layer = tf.keras.layers.Dense(10)(x)
    _output = tf.keras.layers.Dense(1, activation="sigmoid")(crit_hidden_layer)
    _class_output = tf.keras.layers.Dense(n_classes, activation="sigmoid")(x)

    model = tf.keras.Model(_input, [_output, _class_output], name="global_discriminator")
    early_predictor = tf.keras.Model(_input, x, name="early_discriminator")

    return model


def local_discriminator_part(_input, n_classes:int):

    x = tf.keras.layers.Conv1D(16, 5, 2, padding='same')(_input)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Flatten()(x)
    stage1_dropouted = tf.keras.layers.Dropout(0.25)(x)

    _output = tf.keras.layers.Dense(1, activation="sigmoid")(stage1_dropouted)
    # _class_output = layers.Dense(n_classes, activation="sigmoid")(stage1_dropouted)

    return _output


def create_local_discriminator(n_signals:int, sequence_length:int, n_styles:int):
    sig_inputs = tf.keras.Input((sequence_length, n_signals))
    splited_inputs = tf.split(sig_inputs, n_signals, axis=-1)

    crit_outputs = []

    for sig_input in splited_inputs:
        crit_output = local_discriminator_part(sig_input, n_styles)
        crit_outputs.append(crit_output)

    crit_outputs = tf.keras.layers.concatenate(crit_outputs, axis=-1, name="crit_output")

    return tf.keras.Model(sig_inputs, crit_outputs)



##
# Content Encoder
##

def make_content_encoder(seq_length:int, n_feat:int, feat_wiener:int):
    init = tf.keras.initializers.RandomNormal(seed=42)

    _input = tf.keras.Input((seq_length, n_feat))

    x = tf.keras.layers.Conv1D(32, 5, 2, padding='same', kernel_initializer=init)(_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
        
    # x = tf.keras.layers.Conv1D(64, 5, 1, padding='same', kernel_initializer=init)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv1D(64, 5, 2, padding='same', kernel_initializer=init)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv1D(feat_wiener, 5, 1, padding='same', kernel_initializer=init, activation="linear")(x)

    model = tf.keras.Model(_input, x)
    return model

##
# Style Encoder
##

def make_style_encoder(seq_length:int, n_feat:int, vector_output_shape:int):
    init = tf.keras.initializers.RandomNormal(seed=42)

    _input = tf.keras.Input((seq_length, n_feat))

    x = tf.keras.layers.Conv1D(32, 5, 2, padding='same', kernel_initializer=init)(_input) 
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # x = tf.keras.layers.Conv1D(64, 5, 1, padding='same', kernel_initializer=init)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1D(64, 5, 2, padding='same', kernel_initializer=init)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # x = tf.keras.layers.Conv1D(256, 5, 1, padding='same', kernel_initializer=init)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(128, activation=None)(x)
    # x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(vector_output_shape, activation="linear")(x)

    model = tf.keras.Model(_input, x)
    return model
