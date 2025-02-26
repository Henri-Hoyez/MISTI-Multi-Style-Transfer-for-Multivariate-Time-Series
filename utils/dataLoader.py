import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_dataframe(df_path:str, drop_labels=True):
    _df= pd.read_hdf(df_path).astype(np.float32)

    if drop_labels == True:
        _df = _df.drop(columns=['labels'])

    return _df


def train_valid_split(df:pd.DataFrame):
    train_size = 0.8
    train_last_sample = int(df.shape[0]* train_size)
    _df = df.copy()

    train_df = _df.iloc[:train_last_sample]
    valid_df = _df.iloc[train_last_sample:]

    return train_df, valid_df


def is_nan(sequence):
    return tf.reduce_sum(
        tf.cast(
            tf.math.is_nan(sequence),
            tf.int32,
        )
    ) == tf.constant(0)


def pd2tf(df:pd.DataFrame, sequence_lenght, granularity, overlap, batch_size, shuffle:bool):
    total_seq_len = int(sequence_lenght* granularity)
    shift_between_sequences = int(total_seq_len* overlap)

    dset = tf.data.Dataset.from_tensor_slices(df.values)
    dset = dset.window(sequence_lenght , shift=shift_between_sequences, stride=granularity).flat_map(lambda x: x.batch(sequence_lenght, drop_remainder=True))

    dset = dset.filter(is_nan).cache()

    if shuffle == True:
        dset = dset.shuffle(20000)

    if batch_size > 0:
        dset = dset.batch(batch_size, drop_remainder=True)

    return dset.prefetch(100).cache()


def remove_format(path:str):
    return ".".join(path.split('.')[:-1])

def loading_wrapper(df_path:str, sequence_lenght:int, granularity:int, overlap:int, batch_size:int, shuffle:bool=True, drop_labels:bool=True):
    
    path_placeholder = remove_format(df_path)
    
    train_path = f"{path_placeholder}_train.h5"
    valid_path = f"{path_placeholder}_valid.h5"
    
    if '.npy' in df_path:
        print("[+] Load Numpy!")
                
        train_path = train_path.replace(".h5", ".npy")
        valid_path = valid_path.replace(".h5", ".npy")
        
        train_data = load_numpy(train_path)
        valid_data = load_numpy(valid_path)
        
        train_data = np.concat([train_data, train_data], axis=-1)
        valid_data = np.concat([valid_data, valid_data], axis=-1)
    
        train_data = tf.data.Dataset.from_tensor_slices(train_data)
        valid_data = tf.data.Dataset.from_tensor_slices(valid_data)
        
        if batch_size > 0:
            train_data = train_data.batch(batch_size, drop_remainder=True)
            valid_data = valid_data.batch(batch_size, drop_remainder=True)
            
        train_data = train_data.prefetch(batch_size).cache()
        valid_data = valid_data.prefetch(batch_size).cache()
        
        return train_data, valid_data
    
    _df_train = load_dataframe(train_path, drop_labels)
    _df_valid = load_dataframe(valid_path, drop_labels)

    _dset_train = pd2tf(_df_train, sequence_lenght, granularity, overlap, batch_size, shuffle)
    _dset_valid = pd2tf(_df_valid, sequence_lenght, granularity, overlap, batch_size, False)

    return _dset_train, _dset_valid


def get_batches(dset, n_batches):
    _arr = np.array([c for c in dset.take(n_batches)])
    return _arr.reshape((-1, _arr.shape[-2], _arr.shape[-1]))

def get_content_from_numpy(data:np.ndarray):
    content_sequences = []
    for i in range(4):
        content_sequences.append(data[i])
        
    #     plt.figure(figsize=(18, 10))
    #     plt.plot(data[i])
    #     plt.savefig(f"{i}.png")
        
    # exit()
        
    return content_sequences

def load_content_from_unsupervised(dataset_path:str, training_params):
    
    _, dset_valid = loading_wrapper(dataset_path, 
                           training_params.sequence_lenght_in_sample, 
                           training_params.granularity, 
                           training_params.overlap, 
                           5, 
                           False,
                           False)
    
    content_sequences = []
    
    for i, sequence in enumerate(dset_valid.take(5)):
        content_sequences.append(sequence)
                
        plt.figure(figsize=(18, 10))
        plt.plot(sequence[0], ".-")
        plt.savefig(f"{i}.png")
        
    return content_sequences
        

def get_seed_visualization_content_sequences(content_path:str, task_arguments):
    path_placeholder = remove_format(content_path)
    valid_path = f"{path_placeholder}_valid.h5"
    
    if ".npy" in content_path:
        print("[+] Load numpy.")
        valid_path = valid_path.replace(".h5",".npy")
        valid_data = load_numpy(valid_path)
        
        valid_data = np.concat([valid_data, valid_data], axis=-1)
        
        return get_content_from_numpy(valid_data)
    
    _df_valid = load_dataframe(valid_path, task_arguments.unsupervised)
        
    if not task_arguments.unsupervised:
        labels = _df_valid['labels'].unique()
    else:
        return load_content_from_unsupervised(content_path, task_arguments)
    
    content_sequences = []
    
    for l in labels:
        df_part = _df_valid[_df_valid["labels"] == l]
        
        indexes = df_part.index
        
        start_index = indexes[0]
        end_index= start_index+ task_arguments.sequence_lenght_in_sample* task_arguments.granularity

        content_sequence = _df_valid.loc[start_index: end_index-1].values[:, :-1]
        
        content_sequence = content_sequence[::task_arguments.granularity]
        
        # print(content_sequence.shape)
        # exit()
        
        
        content_sequences.append(content_sequence)
                        
    #     plt.figure(figsize=(18, 10))
    #     plt.plot(content_sequence, ".-")
    #     plt.savefig(f"{l}.png")
    # exit()    
    return content_sequences

def load_numpy(data_path:str):
    data = np.load(data_path)
    
    shape = data.shape
    data = data.reshape((shape[0], shape[-1], shape[-2]))
    
    return data