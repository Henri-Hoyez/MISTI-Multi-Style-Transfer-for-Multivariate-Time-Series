import pandas as pd
import numpy as np
import tensorflow as tf
from configs.SimulatedData import Proposed
from dataset.tf_pipeline import make_train_valid_dset

config= Proposed()


def train_valid_split(df, train_size:float=.7):
    dset_size = df.shape[0]
    train_index = int(dset_size* train_size)

    train_split = df.loc[:train_index]
    valid_split = df.loc[train_index:]

    return train_split, valid_split


def pd_to_tf_dset(df_path:str, train_batch_size:int=config.batch_size, valid_batch_size:int=config.valid_set_batch_size):
    _df= pd.read_hdf(df_path).astype(np.float32)
    _df = _df.drop(columns=['labels'])

    content_train, content_valid = make_train_valid_dset(
        _df, 
        config.sequence_lenght_in_sample, 
        config.granularity, 
        int(config.overlap* config.sequence_lenght_in_sample),
        train_batch_size,
        valid_batch_size,
        reduce_train_set=config.reduce_train_set
    )

    return content_train, content_valid


def make_style_dataset(style_dataset_paths:list, train_batch_size:int=config.batch_size, valid_batch_size:int=config.valid_set_batch_size):
    style_train_datasets, style_valid_datasets = [], []

    for s_i, style_path in enumerate(style_dataset_paths):
        dset_style_train, dset_style_valid = pd_to_tf_dset(style_path, train_batch_size)

        dset_style_train = dset_style_train.map(lambda batch: (batch, tf.zeros(batch.shape[0])+ s_i)).cache()
        dset_style_valid = dset_style_valid.map(lambda batch: (batch, tf.zeros(batch.shape[0])+ s_i)).cache()

        dset_style_train = dset_style_train.unbatch()
        dset_style_valid = dset_style_valid.unbatch()
    
        style_train_datasets.append(dset_style_train)
        style_valid_datasets.append(dset_style_valid)

    style_dset_train = tf.data.Dataset.sample_from_datasets(style_train_datasets).batch(train_batch_size, drop_remainder=True)
    style_dset_valid = tf.data.Dataset.sample_from_datasets(style_valid_datasets).batch(valid_batch_size, drop_remainder=True)

    return style_dset_train, style_dset_valid
