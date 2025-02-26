import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv1D, LeakyReLU, Flatten, Dropout, Dense, MaxPool1D
from tensorflow.python.keras.initializers import RandomNormal
from keras.src.layers.normalization.batch_normalization import BatchNormalization

from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy, Accuracy
from tensorflow.python.keras import Sequential


def make_content_space_classif(n_sample_wiener, feat_wiener, n_labels) -> Model:
    init = RandomNormal(seed=42)

    _input = Input((n_sample_wiener, feat_wiener))

    x = Conv1D(16, 3, 1, padding='same', kernel_initializer=init)(_input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)

    x = Dense(n_labels)(x)
    
    model = Model(_input, x)


    model.compile(
        optimizer=Adam(),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=SparseCategoricalAccuracy()
    )

    return model


def make_style_space_classifier(style_vector_shape:tuple, n_labels:int):

    _input = tf.keras.Input((style_vector_shape,))

    x = tf.keras.layers.Dense(n_labels)(_input)
    
    model = tf.keras.Model(_input, x)

    model.compile(
        optimizer=Adam(),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=SparseCategoricalAccuracy()
        )

    return model


def make_real_fake_classifier(seq_shape:tuple)-> Model:
    # initializer = tf.keras.initializers.GlorotNormal()

    model = Sequential()

    model.add(Input(shape=seq_shape))

    model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation='relu'))
    model.add(MaxPool1D(pool_size=2))

    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation='relu'))
    model.add(MaxPool1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(units=50, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(),
        loss=BinaryCrossentropy(from_logits=False),
        metrics=Accuracy()
    )

    return model