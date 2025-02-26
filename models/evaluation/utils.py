import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf
import argparse

from configs.DefaultArguments import DafaultArguments as args

from models.Layers.AdaIN import AdaIN, Moments
from models.backbone.StyleEncoder import NormLayer
from utils import dataLoader, simple_metric, eval_methods
import numpy as np
import json
from models.evaluation import eval_classifiers

from tensorflow.python.keras.models import load_model, Model
from tensorflow.keras import Model

def get_style_name_from_path(path:str):
    filename = path.split("/")[-1]
    return ".".join(filename.split('.')[:-1])


def parse_arguments():
    default_args = args()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp_folder",
        help="the folder where the experiments are located.",
        default=default_args.experiment_folder
    )

    parser.add_argument(
        "--exp_name", 
        help='The name of the experiment ;D', 
        default=default_args.exp_name
    )

    arguments = parser.parse_args()

    return arguments

def load_models(root_folder:str):
    content_encoder = tf.keras.models.load_model(f"{root_folder}/content_encoder.h5")
    style_encoder = tf.keras.models.load_model(f"{root_folder}/style_encoder.h5", custom_objects={"NormLayer":NormLayer})
    decoder = tf.keras.models.load_model(f"{root_folder}/decoder.h5", custom_objects={"Moments":Moments, 'AdaIN':AdaIN})

    return content_encoder, style_encoder, decoder

def load_valid_batches(model_args:dict):
    default_parameters = args()

    sequence_length = default_parameters.simulated_arguments.sequence_lenght_in_sample
    gran = default_parameters.simulated_arguments.granularity
    overlap = default_parameters.simulated_arguments.overlap
    bs = 64
    _, content_dset_valid = dataLoader.loading_wrapper(
        model_args['dset_content'],
        sequence_length, 
        gran, 
        overlap, 
        bs)
    
    _, style1_dset_valid =  dataLoader.loading_wrapper(
        model_args["dset_style_1"], 
        sequence_length, 
        gran, 
        overlap,
        bs)
    
    _, style2_dset_valid =  dataLoader.loading_wrapper(
        model_args["dset_style_2"], 
        sequence_length, 
        gran, 
        overlap,
        bs)
    
    content_batch = dataLoader.get_batches(content_dset_valid, 500)
    style1_batch = dataLoader.get_batches(style1_dset_valid, 500) 
    style2_batch = dataLoader.get_batches(style2_dset_valid, 500) 
    
    return content_batch, style1_batch, style2_batch

def generate(
        content_batch, 
        style_batch, 
        content_encoder, 
        style_encoder, 
        decoder):
    content = content_encoder(content_batch, training=False)
    style = style_encoder(style_batch, training=False)
    generated = decoder([content, style], training=False)
    generated = tf.concat(generated, -1)
    return generated

def encode_dataset(dset:tf.data.Dataset, content_extractor:Model, args:dict) -> tf.data.Dataset:
    label_idx = int(args.simulated_arguments.sequence_lenght_in_sample//2)
    content_space_dataset = dset.map(lambda seq: (content_extractor(seq[:, :, :-1]), seq[:, label_idx, -1]))
    return content_space_dataset

def extract_labels(dset, training_params) -> tf.data.Dataset:
    sequence_length = training_params['sequence_lenght_in_sample']
    idx = sequence_length//2

    return dset.map(lambda seq: (seq[:, :, :-1], seq[:, idx, -1]))

def load_dset(df_path:str, training_params:dict, drop_labels=False, bs = 64) -> tf.data.Dataset:
    sequence_length = training_params['sequence_lenght_in_sample']
    gran = training_params["granularity"]
    overlap = training_params['overlap']
    
    return dataLoader.loading_wrapper(df_path, sequence_length, gran, overlap, bs, drop_labels=drop_labels)

def translate_labeled_dataset(
        content_dset:tf.data.Dataset, 
        style_dset:tf.data.Dataset, 
        content_encoder, 
        style_encoder, 
        decoder) -> tf.data.Dataset:
    
    def __generate(content_batch, style_batch):
        labels= content_batch[1]
        content_space = content_encoder(content_batch[0])
        style_space = style_encoder(style_batch[0])
        
        return tf.concat(decoder([content_space, style_space]), -1), labels
        
    content_style_space = tf.data.Dataset.zip((content_dset, style_dset))
    
    dset_final = content_style_space.map(lambda content_batch, style_batch: (__generate(content_batch, style_batch)))

    return dset_final


def translate_dataset(
        content_dset:tf.data.Dataset, 
        style_dset:tf.data.Dataset, 
        content_encoder, 
        style_encoder, 
        decoder) -> tf.data.Dataset:
    
    content_space = content_dset.map(lambda seq: (content_encoder(seq)))
    style_space = style_dset.map(lambda seq: style_encoder(seq))

    content_style = tf.data.Dataset.zip(content_space, style_space)

    return content_style.map(lambda c,s: tf.concat(decoder([c,s], training=False), -1))
    

def predictions_on_content_space(content_encoder:Model, model_config:dict, data_loading_arguments:dict):

    dset_content_train, dset_content_valid = load_dset(model_config["dset_content"], data_loading_arguments, drop_labels=False)

    dset_style1_train, dset_style1_valid = load_dset(model_config["dset_style_1"], data_loading_arguments, drop_labels=False)

    dset_style2_train, dset_style2_valid = load_dset(model_config["dset_style_2"], data_loading_arguments, drop_labels=False)
    
    dset_content_train= encode_dataset(dset_content_train, content_encoder, data_loading_arguments)
    dset_content_valid= encode_dataset(dset_content_valid, content_encoder, data_loading_arguments)

    dset_style1_train= encode_dataset(dset_style1_train, content_encoder, data_loading_arguments)
    dset_style1_valid= encode_dataset(dset_style1_valid, content_encoder, data_loading_arguments)

    dset_style2_train= encode_dataset(dset_style2_train, content_encoder, data_loading_arguments)
    dset_style2_valid= encode_dataset(dset_style2_valid, content_encoder, data_loading_arguments)

    content_perf = eval_methods.predictions_on_content_space(dset_content_train, dset_content_valid, data_loading_arguments)
    style1_perf = eval_methods.predictions_on_content_space(dset_style1_train, dset_style1_valid, data_loading_arguments)
    style2_perf = eval_methods.predictions_on_content_space(dset_style2_train, dset_style2_valid, data_loading_arguments)

    return content_perf, style1_perf, style2_perf

def tstr(content_path, style_path, content_encoder, style_encoder, decoder, args):
    content_train, content_valid = load_dset(content_path, args)
    style_train, style_valid = load_dset(style_path, args)

    classif_gen_train = translate_labeled_dataset(content_train, style_train, content_encoder, style_encoder, decoder, args)
    classif_gen_valid = translate_labeled_dataset(content_valid, style_valid, content_encoder, style_encoder, decoder, args)

    class_real_train = extract_labels(style_train, args)
    class_real_valid = extract_labels(style_valid, args)

    print('[+] Train on Real.')
    real_performances = eval_methods.train_naive_discriminator(class_real_train, class_real_valid, args)

    print("[+] Train on Synthetic.")
    gen_performances = eval_methods.train_naive_discriminator(classif_gen_train, classif_gen_valid, args)
    return real_performances, gen_performances

def get_model_training_arguments(root_folder):
    with open(f"{root_folder}/model_config.json") as json_file:
        arguments = json.load(json_file)
    return arguments

def get_name(path:str):
    filename = path.split("/")[-1]
    return ".".join(filename.split('.')[:-1])

def get_path(path:str):
    return "/".join(path.split("/")[:-1])
    
def classification_on_style_space(dataset_path:str, style_encoder:Model, default_args:dict):
    style_vector_size = default_args.simulated_arguments.style_vector_size
    n_labels = 5
    epochs = 15

    dset_train, dset_valid = load_dset(dataset_path, default_args)

    style_train = encode_dataset(dset_train, style_encoder, default_args)
    style_valid = encode_dataset(dset_valid, style_encoder, default_args)

    style_classifier = eval_classifiers.make_style_space_classifier(style_vector_size, n_labels)

    style_classifier.fit(style_train, validation_data=style_valid, epochs=epochs)

    return style_classifier.evaluate(style_valid)[1]
    

def real_fake_classification(
        content_path:str, 
        style_path:str, 
        content_encoder:Model, 
        style_encoder:Model, 
        decoder:Model,
        dset_default_arguments:dict):
    
    seq_len = dset_default_arguments.simulated_arguments.sequence_lenght_in_sample
    n_features = dset_default_arguments.simulated_arguments.n_feature
    seq_shape = (seq_len, n_features)
    epochs = 1

    dset_content_train, dset_content_valid = load_dset(content_path, dset_default_arguments, drop_labels=True)
    dset_style_train, dset_style_valid = load_dset(style_path, dset_default_arguments, drop_labels=True)

    style_translated_train = translate_dataset(dset_content_train, dset_style_train, content_encoder, style_encoder, decoder, dset_default_arguments)
    style_translated_valid = translate_dataset(dset_content_valid, dset_style_valid, content_encoder, style_encoder, decoder, dset_default_arguments)

    # Labelize dataset 
    dset_style_train = dset_style_train.map(lambda seq: (seq, tf.ones(seq.shape[0],)))
    dset_style_valid = dset_style_valid.map(lambda seq: (seq, tf.ones(seq.shape[0],)))

    style_translated_train = style_translated_train.map(lambda seq: (seq, tf.zeros(seq.shape[0],)))
    style_translated_valid = style_translated_valid.map(lambda seq: (seq, tf.zeros(seq.shape[0],)))

    dset_train = tf.data.Dataset.sample_from_datasets((dset_style_train, style_translated_train))
    dset_valid = tf.data.Dataset.sample_from_datasets((dset_style_valid, style_translated_valid))

    # Make the discriminator.
    model = eval_classifiers.make_real_fake_classifier(seq_shape)
    model.fit(dset_train, validation_data=dset_valid, epochs=epochs)

    return model.evaluate(dset_valid)[1]