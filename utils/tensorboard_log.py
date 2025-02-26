import matplotlib.pyplot as plt
import numpy as np
import io

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf

from datetime import datetime

from tensorflow.python.keras.metrics import Mean



class TensorboardLog():
    def __init__(self, log_folder, metric_name_list:list) -> None:

        self.full_path = log_folder

        self.train_summary_writer = tf.summary.create_file_writer(f"{self.full_path}/train")
        self.valid_summary_writer = tf.summary.create_file_writer(f"{self.full_path}/valid")
        
        self.train_loggers = {}
        self.valid_loggers = {}

        self.intanciate_loggers(metric_name_list)


    def log_train(self, epoch):
        # image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
        # image = tf.expand_dims(image, 0)

        with self.train_summary_writer.as_default():
            for key, value in self.train_loggers.items():
                tf.summary.scalar(key, value.result(), step=epoch)

            # tf.summary.image("Training Generations", image, step=epoch)
            
    def get_mean_metric(self, dict, key):
        values = []

        for c in dict.keys():
            if key in c:
                values.append(dict[c].result())

        return np.mean(values)     


    def log_valid(self, epoch):
        with self.valid_summary_writer.as_default():
            for key, value in self.valid_loggers.items():
                tf.summary.scalar(key, value.result(), step=epoch)


    def log_train_value(self, key, value, step):
        with self.train_summary_writer.as_default():
            tf.summary.scalar(key, value, step=step)


    def log_valid_value(self, key, value, step):
        with self.valid_summary_writer.as_default():
            tf.summary.scalar(key, value, step=step)


    def reset_metric_states(self):
        for _, logger in self.train_loggers.items():
            logger.reset_states()

    def reset_valid_states(self):
        for _, logger in self.valid_loggers.items():
            logger.reset_states()

    def intanciate_loggers(self, metric_names:list):
        for metric_name in metric_names:
             self.train_loggers[metric_name] = Mean(name=metric_name)
             self.valid_loggers[metric_name] = Mean(name=metric_name)


    def print_train(self, epoch, total_epochs, i, total_batch, extra_info:str=""):
        # print(f"\r e:{epoch}/{total_epochs}; {i}/{total_batch}. G_loss {self.met_generator_train.result():0.2f}; Triplet Loss {self.met_triplet_train.result():0.2f}; Disentanglement Loss: {self.met_disentanglement_train.result():0.2f}; Content Loss {self.met_content_encoder_train.result():0.2f} Local D [Crit; Style]: [{self.met_channel_d_train.result():0.2f}; {self.met_channel_d_style_real_train.result():0.2f}]; Global D [Crit; Style]: [{self.met_central_d_train.result():0.2f}; {self.met_central_d_style_fake_train.result():0.2f}] {extra_info}       ", end="")
        
        g_loss = self.train_loggers["10 - Total Generator Loss"]
        g_crit = self.train_loggers["40 - Global Discriminator Loss"]
        g_classif = self.train_loggers["40 - Global Discriminator Acc"]
        
        c_crit = self.train_loggers['40 - Local Discriminator Loss']
        
        print(f"\r e:{epoch}/{total_epochs}; {i}/{total_batch}. G_loss {g_loss.result():0.2f}; Local D [Crit; Style]: [{c_crit.result():0.2f}]; Global D [Crit; Style]: [{g_crit.result():0.2f}; {g_classif.result():0.2f}] {extra_info}       ", end="")

    def print_valid(self, e, epochs, vb, total_batch, extra_info:str=""):
        
        g_loss = self.valid_loggers["10 - Total Generator Loss"]
        g_crit = self.valid_loggers["40 - Global Discriminator Loss"]
        g_classif = self.valid_loggers["40 - Global Discriminator Acc"]
        
        c_crit = self.valid_loggers['40 - Local Discriminator Loss']
        
        
        
        # print(f"\r e:{e}/{epochs}; {vb}/{total_batch}. G_loss {self.met_generator_valid.result():0.2f}; Triplet Loss {self.met_triplet_valid.result():0.2f}; Disentanglement Loss: {self.met_disentanglement_valid.result():0.2f}; Content Loss {self.met_content_encoder_valid.result():0.2f} Local D [Crit; Style]: [{self.met_channel_d_valid.result():0.2f}; {self.met_channel_d_style_real_valid.result():0.2f}]; Global D [Crit; Style]: [{self.met_central_d_valid.result():0.2f}; {self.met_central_d_style_fake_valid.result():0.2f}] {extra_info}      ", end="")
        print(f"\r e:{e}/{epochs}; {vb}/{total_batch}. G_loss {g_loss.result():0.2f};  Local D [Crit]: [{c_crit.result():0.2f}]; Global D [Crit; Style]: [{g_crit.result():0.2f}; {g_classif.result():0.2f}] {extra_info}      ", end="")

    def fig_to_buff(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf

