import tensorflow as tf
from tensorflow.keras.models import Model
from utils.gpu_memory_grow import gpu_memory_grow
import numpy as np
import argparse
from models.evaluation import utils
from tqdm import tqdm
import os
# os.environ['CUDA_AVAILABLE_DEVICEs'] = "-1"

# print(tf.config.list_physical_devices('GPU'))

gpus = tf.config.list_physical_devices('GPU')
gpu_memory_grow(gpus)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "model_folder", 
        type=str, 
        help='The folder where the trained model is saved.'
    )
    
    args = parser.parse_args()
    return args

def make_dataset(path:str, training_args:dict):
    bs = training_args["batch_size"]
    dset_train, dset_valid = utils.load_dset(path, training_args, drop_labels=False, bs=bs)
    
    if training_args["unsupervised"] == False:
        dset_train = utils.extract_labels(dset_train, training_args)
        dset_valid = utils.extract_labels(dset_valid, training_args)
    
    return dset_train, dset_valid
    

def generate_fake_datasets(content_encoder: Model, style_encoder: Model, decoder: Model, training_params:dict):
    content_path = training_params['dset_content']
    style_paths = training_params["style_datasets"]
    _fake_datasets_train = {}
    _fake_datasets_valid = {}
    
    dset_content_train, dset_content_valid = make_dataset(content_path, training_params)
    
    
    # print(next(iter(dset_content_train)))
    # exit()
    
    for style_dataset_path in tqdm(style_paths):
        style_name = utils.get_style_name_from_path(style_dataset_path)
        
        dset_style_train, dset_style_valid = make_dataset(style_dataset_path, training_params)
        
        if training_params['unsupervised'] == True:
            dset_generated_train = utils.translate_dataset(
                dset_content_train, dset_style_train, 
                content_encoder, style_encoder, decoder)
            
            dset_generated_valid = utils.translate_dataset(
                dset_content_valid, dset_style_valid, 
                content_encoder, style_encoder, decoder)
            
        else:
            dset_generated_train = utils.translate_labeled_dataset(
                dset_content_train, dset_style_train,
                content_encoder, style_encoder, decoder)
            
            dset_generated_valid = utils.translate_labeled_dataset(
                dset_content_valid, dset_style_valid, 
                content_encoder, style_encoder, decoder)
        
        _fake_datasets_train[f"{style_name}_train"] = dset_generated_train
        _fake_datasets_valid[f"{style_name}_valid"] = dset_generated_valid
    
    return _fake_datasets_train, _fake_datasets_valid
        
        
def save_dataset(model_folder:str, datasets:dict):
    generation_folder = f"{model_folder}/generated"
    
    os.makedirs(generation_folder, exist_ok=True)
    
    for key, value in datasets.items():
        save_path = f"{generation_folder}/{key}.tfrecords"
        print(f"[+] Save to {save_path}.")

        value.save(save_path)


def convert_to_univariate(dsets:dict):
    _dsets = {}
    for key, values in dsets.items():
        _values = values.unbatch()
        _dsets[key] = _values.map(lambda seq: tf.convert_to_tensor([seq[:, 0]]))

    return _dsets


def generate(model_folder:str):
    
    training_params = utils.get_model_training_arguments(model_folder)
    
    ce, se, de = utils.load_models(model_folder)
        
    dset_fake_train, dset_fake_valid = generate_fake_datasets(ce, se, de, training_params)
    
    
    
    print(training_params)
    
    if training_params['univariate'] == True:
        
        dset_fake_train = convert_to_univariate(dset_fake_train)
        dset_fake_valid = convert_to_univariate(dset_fake_valid)
                
        # If univariate, that means that the dataset is the Style Time Dataset.
        dset_fake_train = {"style_train":dset_fake_train["style_train"]}
        dset_fake_valid = {"style_valid":dset_fake_valid['style_valid']}
        
        # print(next(iter(dset_fake_train["style_train"])).shape)
        # print(next(iter(dset_fake_valid["style_valid"])).shape)
        # print("******")
                
    save_dataset(model_folder, dset_fake_train)
    save_dataset(model_folder, dset_fake_valid)
    
if __name__ == "__main__":
    args = parse_args()
    generate(args.model_folder)