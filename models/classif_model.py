from utils import dataLoader
from models.NaiveClassifier import make_naive_discriminator
from models.NaiveAutoencoder import make_naive_ae
from utils import utils

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import SparseCategoricalAccuracy, MeanAbsoluteError



class ClassifModel():
    def __init__(self, real_content_dset:str, real_style_dataset_path:list, task_arguments:dict, epochs=50):
        
        folder_name = real_style_dataset_path[0].split("/")[-2]
        
        self.classification_model_folder = f"classification_models/{folder_name}"
                
        os.makedirs(self.classification_model_folder + '/training_curves', exist_ok=True)
    
        sequence_length = task_arguments.sequence_lenght_in_sample
        gran = task_arguments.granularity
        overlap = task_arguments.overlap
        bs = task_arguments.batch_size
        seq_shape = (sequence_length, task_arguments.n_feature)
        n_classes = task_arguments.n_classes
        self.unsupervised = task_arguments.unsupervised
        
        # For evaluation of the generation during the training.
        _, self.dset_content_valid = dataLoader.loading_wrapper(real_content_dset, sequence_length, gran, overlap, bs, drop_labels=False)
        self.dset_content_valid = self.get_task(self.dset_content_valid, task_arguments)

        # Load real style datasets.
        self.models = [] # One small classififer per style (I hope!).
        self.valid_set_styles = [] # Save the validation set for later.
        
        for style_path in real_style_dataset_path:
            
            filename = utils.get_name(style_path)
            model_path = f'{self.classification_model_folder}/{filename}.h5'
            
            dset_train, dset_valid = dataLoader.loading_wrapper(style_path, sequence_length, gran, overlap, bs, drop_labels=False)
            
            dset_train = self.get_task(dset_train, task_arguments)
            dset_valid = self.get_task(dset_valid, task_arguments)
            
            self.valid_set_styles.append(dset_valid)
            
            print(f"[+] Training model for style {filename}.")
            
            if not os.path.exists(model_path):
                if not self.unsupervised:
                    trained_model, history = self.train_supervised(dset_train, dset_valid, epochs, seq_shape, n_classes)
                else: 
                    trained_model, history = self.train_unsupervised(dset_train, dset_valid, epochs, seq_shape)
                    
                trained_model.save(model_path)
                self.plot_learning_curves(history, f"{self.classification_model_folder}/training_curves/{filename}.png")
                
            else:
                print(f"[+] Loading '{model_path}'")
                trained_model = load_model(model_path, custom_objects={"mae":tf.keras.metrics.MeanAbsoluteError})
            
            
            self.models.append(trained_model)
            
    def get_task(self, dset:tf.data.Dataset, task_arguments: dict):
        sequence_length = task_arguments.sequence_lenght_in_sample
        if not task_arguments.unsupervised:
            dset = dset.map(lambda seq: (seq[:, :, :-1], seq[:, sequence_length//2, -1]))
        else:
            dset = dset.map(lambda seq: (seq, seq))
            
        return dset
            
                    
         
    def is_already_trained(self, filepath:str):
        return os.path.exists(filepath)
    
    
    def plot_learning_curves(self, history, save_to):
        
        keys = list(history.history.keys())
        
        plt.figure(figsize=(18, 10))    
        
        ax = plt.subplot(211)
        
        plt.plot(history.history[keys[0]], ".-", label=keys[0])
        plt.plot(history.history[keys[2]], ".-", label=keys[2])
        
        ax.grid(True)
        ax.legend()
        
        ax = plt.subplot(212)
        
        plt.plot(history.history[keys[1]], ".-", label=keys[1])
        plt.plot(history.history[keys[3]], ".-", label=keys[3])
        
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True)
        ax.legend()
        
        plt.savefig(save_to)

    
    def train_supervised(self, train_dset, valid_dset, epochs, seq_shape, n_classes):
        model = make_naive_discriminator(seq_shape, n_classes)
        history = model.fit(train_dset, validation_data=valid_dset, epochs=epochs)
        return model, history
    
    
    def train_unsupervised(self, train_dset, valid_dset, epochs, seq_shape):
        model = make_naive_ae(seq_shape)
        history = model.fit(train_dset, validation_data=valid_dset, epochs=epochs)
        return model, history
    
    
    def generate(self, ce:Model, se:Model, de:Model, cont_batch, style_batch):
        content_encodings = ce(cont_batch)
        style_encodings = se(style_batch)
        sequences = de([content_encodings, style_encodings])
        return tf.concat(sequences, -1)
    
    def evaluate(self, content_encoder:Model, style_encoder:Model, decoder:Model):
        print('[+] Classitifation metric.')
        accs = []
        
        if self.unsupervised:
            acc_metric = MeanAbsoluteError()
        else:
            acc_metric = SparseCategoricalAccuracy()
        
        for i in range(len(self.valid_set_styles)):
            
            selected_valid_style = self.valid_set_styles[i]
            selected_model = self.models[i]
            acc_metric.reset_state()
            
            for (content_batch, content_label), (style_batch, _) in zip(self.dset_content_valid.take(100), selected_valid_style.take(100)):
                generated_batch = self.generate(content_encoder, style_encoder, decoder, content_batch, style_batch)
                
                model_pred_on_synth = selected_model(generated_batch)
                
                acc_metric.update_state(content_label, model_pred_on_synth)
                print(f"\r Style {i}; {acc_metric.result().numpy():0.3f}", end="")
                
            accs.append(acc_metric.result().numpy())
        return np.mean(accs)
    
       