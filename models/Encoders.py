import tensorflow as tf


class StyleEncoder(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def build_model(self, sequence_shape:tuple):
        _model = tf.keras.Sequential()
        _model.add(tf.keras.layers.Conv1D(input_shape=sequence_shape, filters=32, kernel_size=3))
        _model.add(tf.keras.layers.BatchNormalization())
        _model.add(tf.keras.layers.Activation('relu'))
        _model.add(tf.keras.layers.MaxPooling1D(2))
        
        _model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3))
        _model.add(tf.keras.layers.BatchNormalization())
        _model.add(tf.keras.layers.Activation('relu'))
        _model.add(tf.keras.layers.MaxPooling1D(2))
        _model.add(tf.keras.layers.Flatten())
        
        return _model
        
        
class ContentEncoder(tf.keras.Model):
    def __init__(self, seq_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model = self.build_content_encoder(seq_shape)
        
    def build_content_encoder(self, sequence_shape:tuple):
        _model = tf.keras.Sequential()
        _model.add(tf.keras.layers.Conv1D(input_shape=sequence_shape, filters=64, kernel_size=3, padding="same"))
        _model.add(tf.keras.layers.BatchNormalization())
        _model.add(tf.keras.layers.Activation('relu'))
        # _model.add(tf.keras.layers.MaxPooling1D(2))
        
        _model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same"))
        _model.add(tf.keras.layers.BatchNormalization())
        _model.add(tf.keras.layers.Activation('relu'))
        # _model.add(tf.keras.layers.MaxPooling1D(2))
        
        _model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="same"))
        _model.add(tf.keras.layers.BatchNormalization())
        _model.add(tf.keras.layers.Activation('relu'))
        # _model.add(tf.keras.layers.MaxPooling1D(2))
        
        _model.add(tf.keras.layers.Conv1D(filters=2, kernel_size=3, padding="same"))
        _model.add(tf.keras.layers.BatchNormalization())
        _model.add(tf.keras.layers.Activation('relu'))
        # _model.add(tf.keras.layers.MaxPooling1D(3))
        
        return _model
        