import numpy as np
from configs.MetricConfig import Metric

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf
from configs.MetricConfig import Proposed
configs = Proposed()


def optimized_cov(a,v):
    _seq_len = a.shape[1]
    nan_mean_a = np.nanmean(a, axis=1).reshape((-1,1))
    nan_mean_b = np.nanmean(v, axis=1).reshape((-1,1))
    return np.nansum((a- nan_mean_a)* (v- nan_mean_b), axis=1)

def mean_difference(a,v):
    return np.nanmean(a) - np.nanmean(v)

def optimized_windowed_cov(a, v, beta=Metric().mean_senssibility_factor):
    if a.shape[1] > v.shape[1]:
        _a, _v = v, a 
    else: 
        _a, _v = a, v

    n = _a.shape[1]
    corrs = []

    aa = optimized_cov(_a, _a)
    for k in range(_v.shape[1] - _a.shape[1]):
        __v = _v[:, k: n+k]
        # Compute the covariance 

        av = optimized_cov(_a,__v)
        vv = optimized_cov(__v, __v)
        _mean_diff = beta* mean_difference(_a,__v)

        augmented_cov = av/np.sqrt(aa*vv)+ _mean_diff
        # /(np.sqrt(aa*vv)) 

        corrs.append(augmented_cov)
        
    return np.array(corrs)

def signature_on_batch(x:np.ndarray, ins:list, outs:list, sig_seq_len:int):
    """Compute the signature from a given batch of MTS sequences `x`

    Args:
        x (np.ndarray): the batch
        ins (list): input columns
        outs (list): output label solumns
        sig_seq_len (int): the desired signature length

    Returns:
        np.ndarray: the min, max, mean signature.
    """
    sigs = []
    shift = sig_seq_len//2
    childrens = x[:, shift:-shift]

    for _in in ins:
        for _out in outs:
            c1 = x[:, :, _in]
            c2 = childrens[:, :, _out]
            
            sig = optimized_windowed_cov(c1, c2)

            sigs.append(sig)

    mins = np.min(sigs, axis=-1)
    maxs = np.max(sigs, axis=-1)
    means= np.mean(sigs, axis=-1)   

    signatures = np.stack([mins, maxs, means], axis=-1)
    
    return signatures

def signature_metric(source_sig:np.ndarray, target_sig:np.ndarray):
    # Shape: (n_features, sign_seq_lenght, 3)

    averaged_source = np.mean(source_sig, axis=0)
    averaged_target = np.mean(target_sig, axis=0)

    min_source = averaged_source[:, 0]
    max_source = averaged_source[:, 1]
    mean_source = averaged_source[:, 2]

    min_target = averaged_target[:, 0]
    max_target = averaged_target[:, 1]
    mean_target = averaged_target[:, 2]

    mean_differences = np.sum(np.abs(mean_target- mean_source))
    mins_differences = np.sum(np.abs(min_target- min_source))
    maxs_differences = np.sum(np.abs(max_target- max_source))

    metric = (mean_differences + mins_differences +maxs_differences)/3

    return metric


def compute_metric(generated_style:tf.Tensor, true_style:tf.Tensor, config:object=configs):
    true_signature = signature_on_batch(true_style, config.met_params.ins, config.met_params.outs, config.met_params.signature_length)
    generated_signature= signature_on_batch(generated_style, config.met_params.ins, config.met_params.outs, config.met_params.signature_length)

    return signature_metric(true_signature, generated_signature)


def step_fonction(xs, treashold=0.5):
    return tf.cast(tf.math.greater_equal(xs, treashold), tf.float32)


def accuracy(y_true, y_pred):
    y_pred = step_fonction(y_pred)

    y_pred = tf.reshape(y_pred, (-1,))
    y_true = tf.reshape(y_true, (-1,))
    
    good_prediction = tf.reduce_sum(tf.cast(y_pred == y_true, tf.float32))

    return good_prediction/y_true.shape[0]