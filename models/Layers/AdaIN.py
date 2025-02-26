
# import os
# os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.ops import moments, sqrt



class Moments(Layer):
    # def __init__(self, *args, **kwargs):
    #     super(Moments, self).__init__(*args, **kwargs)
    
    def call(self, x):
        return tf.keras.ops.moments(x, axes=[1], keepdims=True)
    
    
    
class Sqrt(Layer):
    def __init__(self, *args, **kwargs):
        super(Sqrt, self).__init__(*args, **kwargs)   
        
    def call(self, x):
        return sqrt(x)


# Define AdaIN Layers for Time Series
class AdaIN(Layer):
    def __init__(self, *args, **kwargs):
        super(AdaIN, self).__init__(*args, **kwargs)

    def get_mean_std(self, x, eps=1e-5):
        # _mean, _variance = moments(x, axes=[1], keepdims=True)
        # standard_dev = sqrt_layer()(_variance+ eps)
        _mean, _variance = Moments()(x)
        
        standard_dev = Sqrt()(_variance+ eps)
        
        return _mean, standard_dev

    def call(self, content_input, style_input):
        # print(content_input.shape, style_input.shape)
        content_mean, content_std = self.get_mean_std(content_input)
        style_mean, style_std = self.get_mean_std(style_input)
        adain_res =style_std* (content_input - content_mean) / content_std+ style_mean
        return adain_res