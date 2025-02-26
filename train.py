### Train the first version of the time series style transfer.

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ["TF_USE_LEGACY_KERAS"]="1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# from keras import __version__
# tf.keras.__version__ = __version__

import logging
import argparse

from utils.gpu_memory_grow import gpu_memory_grow
from configs.DefaultArguments import DafaultArguments as args
from utils import dataLoader
from algorithms.mts_style_transferv2 import Trainer
from configs.get_data_config import get_dataset_class


def parse_arguments():
    default_args = args()
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--task", 
        help='Choice of the task (e.g. InputNoise, OutputNoise, TimeShift, CausalShift, HAR, GoogleStocks, EnergyAppliance).',
        default="InputNoise"
    )
    
    parser.add_argument(
        "--epochs",
        help='Number of epochs', type=int,
        default=default_args.epochs
        )

    parser.add_argument(
        "--tensorboard_root", 
        help='The root folder for tensorflow', 
        default=default_args.tensorboard_root_folder
    )
    
    parser.add_argument(
        "--exp_folder", 
        help='Helpfull for grouping experiment', 
        default=default_args.experiment_folder
    )

    parser.add_argument(
        "--exp_name", 
        help='The name of the experiment ;D', 
        default=default_args.exp_name
    )
    
    parser.add_argument(
        '--save_to', 
        help='The folder where the model will be saved.', 
        default=default_args.default_root_save_folder
    )
    
    parser.add_argument(
        "--cpu",
        help='If set the algorithm will run on cpu',
        action='store_true'
    )
    
    parser.add_argument("--restore_from", type=str, default=None)
    
    arguments = parser.parse_args()

    return arguments


def remove_format(path:str):
    return ".".join(path.split('.')[:-1])


def main():
    shell_arguments = parse_arguments()

    task_arguments = get_dataset_class(shell_arguments.task)()
    
    if shell_arguments.cpu:
        print("[+] Execute training on CPU.")
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    gpu_memory_grow(gpus)
    
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR) 
    

    sequence_length = task_arguments.sequence_lenght_in_sample
    gran = task_arguments.granularity
    overlap = task_arguments.overlap
    bs = task_arguments.batch_size
    
    ###
    content_viz_sequences = dataLoader.get_seed_visualization_content_sequences(
        task_arguments.content_dataset, 
        task_arguments
    )
                    
    content_dset_train, content_dset_valid = dataLoader.loading_wrapper(
        task_arguments.content_dataset,
        sequence_length, 
        gran, 
        overlap, 
        2*bs, drop_labels=not task_arguments.unsupervised) # Two Times BS for the training function.
    
    # Load Styles:
    style_dsets_train, style_dsets_valid = [], []
    style_seeds_train, style_seeds_valid = [], []
    
    for i, style_path in enumerate(task_arguments.style_datasets):
        style_labels = tf.zeros((1,)) + i

        style_train, style_valid =  dataLoader.loading_wrapper(
            style_path, 
            sequence_length, 
            gran, 
            overlap,
            0, drop_labels=not task_arguments.unsupervised
        )
                        
        _style_seed_train = dataLoader.get_batches(style_train.batch(bs), 2)
        _style_seed_valid = dataLoader.get_batches(style_valid.batch(bs), 2)
        
        style_seeds_train.append(_style_seed_train)
        style_seeds_valid.append(_style_seed_valid)

        style_train = style_train.map(lambda seq: (seq, style_labels))
        style_valid = style_valid.map(lambda seq: (seq, style_labels))
        
        style_dsets_train.append(style_train)
        style_dsets_valid.append(style_valid)
    
    style_seeds_train = tf.convert_to_tensor(style_seeds_train)
    style_seeds_valid = tf.convert_to_tensor(style_seeds_valid)

    style_dsets_train = tf.data.Dataset.sample_from_datasets(style_dsets_train).batch(bs, drop_remainder=True).prefetch(100).cache()
    style_dsets_valid = tf.data.Dataset.sample_from_datasets(style_dsets_valid).batch(bs, drop_remainder=True).prefetch(100).cache()

    trainner = Trainer(shell_arguments, task_arguments)

    trainner.instanciate_datasets(
        content_dset_train, content_dset_valid,
        style_dsets_train, style_dsets_valid,
    )

    trainner.set_seeds(style_seeds_train, style_seeds_valid, content_viz_sequences)

    trainner.train()


if __name__ == "__main__":
    main()