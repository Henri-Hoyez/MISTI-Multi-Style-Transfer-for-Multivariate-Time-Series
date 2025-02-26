import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from configs.get_data_config import get_dataset_class
from utils.gpu_memory_grow import gpu_memory_grow

import matplotlib.pyplot as plt
from models.evaluation import utils
from utils import eval_methods, simple_metric

from utils import metric
from utils import prd

import argparse


gpus = tf.config.list_physical_devices('GPU')
gpu_memory_grow(gpus)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_folder", 
        help='The folder where the trained model is saved.', 
    )
    
    parser.add_argument("--epochs", type=int, default=20, help='Epochs to train the TSTR model on.')
    
    return parser.parse_args()

def get_name(path:str):
    filename = path.split("/")[-1]
    return ".".join(filename.split('.')[:-1])


def get_domain_shift_from_file(path: str):
    return path.split("/")[2]


def load_generations(model_folder:list, domain_shift_type:str):
    generations = {}
    print(f"{model_folder}/{domain_shift_type}")
    for root, folders, files in os.walk(f"{model_folder}"):
        # print(files)
        # print(folders)
        for f in folders:
            if ".tfrecords" in f:
                print(f"[+] loading Fake: {f}")
                key = get_name(f)
                data = tf.data.Dataset.load(f"{root}/{f}")
                
                shape = next(iter(data))[0].shape

                if len(shape) < 3:
                    data = data.batch(64)

                
                generations[key] = data
                
    return generations

def load_style_dataset(training_params):
    style_dataset = {}
    bs = 128
    
    style_paths = training_params['style_datasets']
    
    for path in style_paths:
        key = get_name(path)
        print(f"[+] Load Real: {key}")
        
        style_train, style_valid = utils.load_dset(path, training_params, drop_labels=False, bs=bs)
        
        style_train = utils.extract_labels(style_train, training_params)
        style_valid = utils.extract_labels(style_valid, training_params)
        
        style_dataset[f"{key}_train"] = style_train
        style_dataset[f"{key}_valid"] = style_valid
        
    return style_dataset
    
    
def get_batches(dset, n_batches):
    _arr = np.array([c[0] for c in dset.take(n_batches)])
    
    return _arr.reshape((-1, _arr.shape[-2], _arr.shape[-1]))
 
def compute_metrics(dset_real, dset_fake, style_names, model_folder: str):
    def time_shift_evaluation(big_batch):
        return [simple_metric.estimate_time_shift(big_batch, 0, i) for i in range(big_batch.shape[-1])]
    
    real_noise_metric, gen_noise_metric = [], []
    real_ampl_metric, gen_ampl_metric = [], []
    real_ts_metric, gen_ts_metric = [], []
    proposed_metric = []

    for style_name in style_names:
        print(f"[+] Compute metric for {style_name}")
        real_batch = get_batches(dset_real[f"{style_name}_valid"], 5)
        fake_batch = get_batches(dset_fake[f"{style_name}_valid"], 5)
        
        real_noise_metric.append(simple_metric.simple_metric_on_noise(real_batch)[-1])
        gen_noise_metric.append(simple_metric.simple_metric_on_noise(fake_batch)[-1])
        
        real_ampl_metric.append(simple_metric.extract_amplitude_from_signals(real_batch))
        gen_ampl_metric.append(simple_metric.extract_amplitude_from_signals(fake_batch))
        
        real_ts_metric.append(time_shift_evaluation(real_batch))
        gen_ts_metric.append(time_shift_evaluation(fake_batch))
        
        metric_on_style = metric.compute_metric(fake_batch, real_batch)
        
        proposed_metric.append(metric_on_style)     
        
    real_mean_noises = np.mean(real_noise_metric, axis=-1).reshape((-1, 1))
    fake_mean_noises = np.mean(gen_noise_metric, axis=-1).reshape((-1, 1))
    mean_noises = np.concatenate((real_mean_noises, fake_mean_noises), axis=-1)
    
    real_mean_ampl = np.mean(real_ampl_metric, axis=-1).reshape((-1, 1))
    fake_mean_ampl = np.mean(gen_ampl_metric, axis=-1).reshape((-1, 1))
    mean_ampl= np.concatenate((real_mean_ampl, fake_mean_ampl), axis=-1)
    
    real_mean_time_shift = np.mean(real_ts_metric, axis=-1).reshape((-1, 1))
    fake_mean_time_shift = np.mean(gen_ts_metric, axis=-1).reshape((-1, 1))
    mean_time_shift= np.concatenate((real_mean_time_shift, fake_mean_time_shift), axis=-1)
    
    df_noises = pd.DataFrame(data=mean_noises, index=style_names, columns=['Real', 'Fake'])
    df_ampl = pd.DataFrame(data=mean_ampl, index=style_names, columns=['Real', 'Fake'])
    df_time_shift = pd.DataFrame(data=mean_time_shift, index=style_names, columns=['Real', 'Fake'])
    
    df_proposed_metric = pd.DataFrame(data=[proposed_metric], columns=style_names)
    
    df_noises.to_excel(f'{model_folder}/noise_metric.xlsx')
    df_ampl.to_excel(f'{model_folder}/ampl_metric.xlsx')
    df_time_shift.to_excel(f'{model_folder}/time_shift_metric.xlsx')
    df_proposed_metric.to_excel(f'{model_folder}/metric_proposed.xlsx')
    
    return df_noises, df_ampl, df_time_shift 



def tstr(
    dset_train_real,
    dset_valid_real,
    dset_train_fake, 
    dset_valid_fake, 
    save_to:str,
    training_params:dict,
    epochs = 50):
    
    n_classes = training_params['n_classes']
    
    print("[+] Train Synthetic, Test Real")
    tstr_perfs, tstr_hist, _ = eval_methods.train_naive_discriminator(dset_train_fake, dset_valid_real, training_params, epochs=epochs, n_classes=n_classes)
    
    print("[+] Train Real, Test Synthetic")
    _, trts_hist, _ = eval_methods.train_naive_discriminator(dset_train_real, dset_valid_fake, training_params, epochs=epochs, n_classes=n_classes)
    
    print('[+] Train Real, Test Real.')
    trtr_perfs, trtr_hist, embbeding_model = eval_methods.train_naive_discriminator(dset_train_real, dset_valid_real, training_params, epochs=epochs, n_classes=n_classes)

    print("[+] Train Synthetic, Test Synthetic")
    _, tsts_hist, _ = eval_methods.train_naive_discriminator(dset_train_fake, dset_valid_fake, training_params, epochs=epochs, n_classes=n_classes)
    
    fig = plt.figure(figsize=(18, 10))
    
    ax = plt.subplot(421)
    ax.set_title("Train Real Test Real loss")
    
    plt.plot(trtr_hist.history["loss"], ".-", label='Train')
    plt.plot(trtr_hist.history["val_loss"], ".-", label='Valid')
    ax.grid()
    ax.legend()
    ax.set_ylim(0, 1.)
    
    ax = plt.subplot(422)
    ax.set_title("Train Real Test Real accuracy")
    
    plt.plot(trtr_hist.history["sparse_categorical_accuracy"], ".-", label='Train')
    plt.plot(trtr_hist.history["val_sparse_categorical_accuracy"], ".-", label='Valid')
    ax.grid()
    ax.legend()
    ax.set_ylim(0, 1.)
    
    
    #######
    ax = plt.subplot(423)
    ax.set_title("Train Real, Test Synthetic loss")
    
    plt.plot(trts_hist.history["loss"], ".-", label='Train Real, Test Synthetic (Train)')
    plt.plot(trts_hist.history["val_loss"], ".-", label='Train Real, Test Synthetic (Valid)')
    
    ax.grid()
    ax.legend()

    ax = plt.subplot(424)
    ax.set_title("Train Real, Test Synthetic accuracy")
    
    plt.plot(trts_hist.history["sparse_categorical_accuracy"], ".-", label='Train Real, Test Synthetic (Train)')
    plt.plot(trts_hist.history["val_sparse_categorical_accuracy"], ".-", label='Train Real, Test Synthetic (Valid)')
    
    ax.grid()
    ax.legend()
    #######
    
    ax = plt.subplot(425)
    ax.set_title("Train Synthetic, Test Synthetic loss")
    
    plt.plot(tsts_hist.history["loss"], ".-", label='Train')
    plt.plot(tsts_hist.history["val_loss"], ".-", label='Valid')
    ax.grid()
    ax.legend()
    
    ax = plt.subplot(426)
    ax.set_title("Train Synthetic, Test Synthetic accuracy")
    
    plt.plot(tsts_hist.history["sparse_categorical_accuracy"], ".-", label='Train')
    plt.plot(tsts_hist.history["val_sparse_categorical_accuracy"], ".-", label='Valid')
    ax.grid()
    ax.legend()
    #######
    
    ax = plt.subplot(427)
    ax.set_title("Train Synthetic, Test Real loss")
    
    plt.plot(tstr_hist.history["loss"], ".-", label='Train')
    plt.plot(tstr_hist.history["val_loss"], ".-", label='Valid')
    ax.grid()
    ax.legend()
    
    ax = plt.subplot(428)
    ax.set_title("Train Synthetic, Test Real accuracy")
    
    plt.plot(tstr_hist.history["sparse_categorical_accuracy"], ".-", label='Train')
    plt.plot(tstr_hist.history["val_sparse_categorical_accuracy"], ".-", label='Valid')
    ax.grid()
    ax.legend()
    #######
    
    plt.savefig(save_to)
    
    plt.close(fig)
    
    # Train Real Test Real, Train Synthetic Test Real, F1 score
    return trtr_perfs, tstr_perfs, None

def tstr_on_styles(real_dataset, fake_dataset, model_folder, style_names, training_params, epochs=50):
    tstr_stats = {}
    f1_stats = {}
        
    for _, style_ in enumerate(style_names):
        print(f'[+] Training on dataset {style_}.')
        
        perf_on_real, perf_on_fake, f1 = tstr(
            real_dataset[f"{style_}_train"],
            real_dataset[f"{style_}_valid"],
            fake_dataset[f"{style_}_train"],
            fake_dataset[f"{style_}_valid"], 
            f'{model_folder}/tstr_{style_}.png', 
            training_params, epochs=epochs
            )
        
        
        
        tstr_stats[f"{style_}_real"] = [perf_on_real]
        tstr_stats[f"{style_}_gen"] = [perf_on_fake]
        f1_stats[f"{style_}"] = [f1]
        
    tstr_stats = pd.DataFrame.from_dict(tstr_stats)
    f1_stats = pd.DataFrame.from_dict(f1_stats)
    
    tstr_stats.to_excel(f"{model_folder}/tstr.xlsx")
    # f1_stats.to_excel(f"{model_folder}/f1s.xlsx")
    
    return tstr_stats, f1_stats     


def plot_signatures(real_styles:dict, generated_styles: dict, style_names: list, task_name: str, model_folder: str):
    
    for key in style_names:
        real_style = real_styles[f"{key}_valid"]
        generated_style = generated_styles[f"{key}_valid"]
        
        real_batch = get_batches(real_style, 5)
        fake_batch = get_batches(generated_style, 5)
        
        task_params = get_dataset_class(task_name)()
        
        ins = task_params.met_params.ins
        outs = task_params.met_params.outs
        signature_length = task_params.met_params.signature_length
    
        # Returns: np.stack([mins, maxs, means], axis=-1)
        # Shape (n_senssors, signature_length, 3)
        real_signature = metric.signature_on_batch(real_batch, ins, outs, signature_length)
        fake_signature = metric.signature_on_batch(fake_batch, ins, outs, signature_length)
        
        t = np.arange(0, signature_length)
        
        plt.figure(figsize=(18, 10))
        plt.title(f"Real Signature compared to model Generation {key}")
        ax = plt.subplot(111)
        ax.grid(True)
        
        r_mins = np.mean(real_signature[:, :, 0], axis=0)
        r_maxs = np.mean(real_signature[:, :, 1], axis=0)
        r_means = np.mean(real_signature[:, :, 2], axis=0)
        
        plt.fill_between(t, r_mins, r_maxs, color='b', alpha=0.5, label="real")
        
        plt.plot(t, r_mins, "b")
        plt.plot(t, r_maxs , "b")
        plt.plot(t, r_means, "b")
        
        f_mins = np.mean(fake_signature[:, :, 0], axis=0)
        f_maxs = np.mean(fake_signature[:, :, 1], axis=0)
        f_means = np.mean(fake_signature[:, :, 2], axis=0)
                
        plt.fill_between(t, f_mins, f_maxs, color='r', alpha=0.5, label='Generated')
        plt.plot(t, f_mins , "r")
        plt.plot(t, f_maxs , "r")
        plt.plot(t, f_means, "r")
        plt.legend()
        
        plt.savefig(f"{model_folder}/{key}_valid.png")
        
    
def main():
    shell_arguments = parse_arguments()

    model_folder = shell_arguments.model_folder
    epochs = shell_arguments.epochs
    
    training_params = utils.get_model_training_arguments(model_folder)
    task = training_params['task']
    ds_type = get_domain_shift_from_file(training_params['style_datasets'][0])
    
    fake_dset = load_generations(model_folder, ds_type)
    
    real_dset = load_style_dataset(training_params)
    
    style_names = [get_name(f) for f in training_params['style_datasets']]
    
    plot_signatures(real_dset, fake_dset, style_names, task, model_folder)
    
    compute_metrics(real_dset, fake_dset, style_names, model_folder)
    
    tstr_on_styles(real_dset, fake_dset, model_folder, style_names, training_params, epochs=epochs)


if __name__ == "__main__":
    main()