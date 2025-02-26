from utils.dataLoader import get_batches
from utils.tensorboard_log import TensorboardLog
from utils import metric, simple_metric, visualization_helpers, MLFlow_utils
from models.classif_model import ClassifModel

from models import losses
import numpy as np
from sklearn.decomposition import PCA

from tensorflow.keras.metrics import BinaryAccuracy, binary_accuracy
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

from keras._tf_keras.keras.utils import plot_model

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf
from datetime import datetime

class Trainer():
    def __init__(self, shell_arguments, task_arguments) -> None:
        
        self.shell_arguments = shell_arguments
        self.task_arguments = task_arguments
        n_styles = len(self.task_arguments.style_datasets)
        
        self.batch_size = task_arguments.batch_size
        self.epochs = shell_arguments.epochs
        
        ### Decoder lambdas 
        self.l_reconstr = task_arguments.l_reconstr
        self.l_global = task_arguments.l_global 
        self.l_local = task_arguments.l_local
        
        #### Global v.s. channel reconstruction loss.
        self.l_global_reconstr = task_arguments.l_global_reconstr
        self.l_local_reconstr = task_arguments.l_local_reconstr
        
        ### Content Encoder lambdas
        self.l_content = task_arguments.l_content
        self.e_adv = task_arguments.encoder_adv
        self.encoder_recontr = task_arguments.encoder_recontr
        
        ### Style encoder lambdas
        self.l_triplet = task_arguments.l_triplet
        self.l_disentanglement = task_arguments.l_disentanglement
        self.style_preservation = task_arguments.l_style_preservation
        
        
        self.content_encoder = task_arguments.get_content_encoder()
        self.decoder = task_arguments.get_generator()
        self.style_encoder = task_arguments.get_style_encoder()
        self.global_discriminator = task_arguments.get_global_discriminator()
        self.local_discriminator = task_arguments.get_local_discriminator()
        
        # self.opt_content_encoder = Adam(learning_rate=self.task_arguments.learning_rate, beta_1=0.5) # 0.0005
        # self.opt_style_encoder = Adam(learning_rate=self.task_arguments.learning_rate, beta_1=0.5) # 0.0005
        # self.opt_decoder = Adam(learning_rate=self.task_arguments.learning_rate, beta_1=0.5) # 0.0005
        # self.local_discriminator_opt = Adam(learning_rate=self.task_arguments.learning_rate, beta_1=0.5) # 0.0005
        # self.global_discriminator_opt = Adam(learning_rate=self.task_arguments.learning_rate, beta_1=0.5) # 0.0005 
        
        self.opt_content_encoder = RMSprop(learning_rate=self.task_arguments.learning_rate)
        self.opt_style_encoder = RMSprop(learning_rate=self.task_arguments.learning_rate)
        self.opt_decoder = RMSprop(learning_rate=self.task_arguments.learning_rate)
        self.local_discriminator_opt = RMSprop(learning_rate=self.task_arguments.learning_rate)
        self.global_discriminator_opt = RMSprop(learning_rate=self.task_arguments.learning_rate)
        
        self.best_metric_value = np.inf
        self.initial_epoch= 0
        
        if not shell_arguments.restore_from is None:
            self.restore_from(shell_arguments.restore_from)
        else:
            self.log_folder = self.get_log_folder()
                
            self.chkpt = tf.train.Checkpoint(
                step=tf.Variable(self.initial_epoch), 
                content_encoder=self.content_encoder,
                style_encoder=self.style_encoder,
                decoder=self.decoder,
                global_discriminator=self.global_discriminator,
                local_discriminator=self.local_discriminator,
                opt_content_encoder= self.opt_content_encoder,
                opt_style_encoder= self.opt_style_encoder, 
                opt_decoder= self.opt_decoder,
                local_discriminator_opt=self.local_discriminator_opt,
                global_discriminator_opt=self.global_discriminator_opt
            )
        
            self.manager = tf.train.CheckpointManager(self.chkpt, f'{self.log_folder}/tf_ckpts', max_to_keep=3)

        # Prepare the classification metric.
        # if not ".npy" in task_arguments.content_dataset:
        #     # univariate dataset from numpy are unsupervised
        #     self.classif_metric = ClassifModel(task_arguments.content_dataset, task_arguments.style_datasets, task_arguments)
        
        self.classif_metric = ClassifModel(task_arguments.content_dataset, task_arguments.style_datasets, task_arguments)
        

        self.prepare_loggers(n_styles)
        self.plot_models()

    def get_log_folder(self):
        try:
            _log_folder = self.log_folder
            
        except AttributeError:
            root = self.shell_arguments.tensorboard_root
            date_str = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
            exp_name = self.shell_arguments.exp_name
            _log_folder = f"{root}/{date_str} - {exp_name}"
            
        return _log_folder



    def plot_models(self):
        plot_model(self.decoder, show_shapes=True, to_file='Decoder.png',  expand_nested=True)
        plot_model(self.global_discriminator, show_shapes=True, to_file='global_discriminator.png',  expand_nested=True)
        plot_model(self.local_discriminator, show_shapes=True, to_file='local_discriminator.png',  expand_nested=True)
        plot_model(self.content_encoder, show_shapes=True, to_file='content_encoder.png',  expand_nested=True)
        plot_model(self.style_encoder, show_shapes=True, to_file='style_encoder.png',  expand_nested=True)

    def restore_from(self, path):
        
        self.chkpt = tf.train.Checkpoint(
                step=tf.Variable(self.initial_epoch), 
                content_encoder=self.content_encoder,
                style_encoder=self.style_encoder,
                decoder=self.decoder,
                global_discriminator=self.global_discriminator,
                local_discriminator=self.local_discriminator,
                opt_content_encoder= self.opt_content_encoder,
                opt_style_encoder= self.opt_style_encoder, 
                opt_decoder= self.opt_decoder,
                local_discriminator_opt=self.local_discriminator_opt,
                global_discriminator_opt=self.global_discriminator_opt
            )        
        
        self.manager = tf.train.CheckpointManager(self.chkpt, f'{path}/tf_ckpts', max_to_keep=3)
        self.chkpt.restore(self.manager.latest_checkpoint)
        
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
            self.initial_epoch = int(self.chkpt.step)
            
            self.log_folder = path
            n_styles = len(self.task_arguments.style_datasets)
            self.prepare_loggers(n_styles) 
            
            
        else:
            print("Initializing from scratch.")


    def generate(self, content_batch, style_batch):
        content = self.content_encoder(content_batch, training=False)
        style = self.style_encoder(style_batch, training=False)
        generated = self.decoder([content, style], training=False)
        generated = tf.concat(generated, -1)
        return generated
    
    
    def train(self):
        total_batch_train = "?"
        total_batch_valid = "?"

        for e in range(self.initial_epoch, self.epochs):
            self.logger.reset_metric_states()
            self.logger.reset_valid_states()
            
            print("[+] Train Step...")  
            for i, (content_batch, style_batch) in enumerate(zip(self.dset_content_train, self.dsets_style_train)):
                content_sequence1 = content_batch[:int(self.batch_size)]
                content_sequence2 = content_batch[int(self.batch_size):]
            
                self.discriminator_step(content_sequence1, style_batch)

                self.generator_step(content_sequence1, content_sequence2, style_batch)

                self.logger.print_train(e, self.epochs, i, total_batch_train)
            
            print()
            print("[+] Validation Step...")
            for vb, (content_batch, style_batch) in enumerate(zip(self.dset_content_valid, self.dsets_style_valid)):
                content_sequence1 = content_batch[:int(self.batch_size)]
                content_sequence2 = content_batch[int(self.batch_size):]

                self.discriminator_valid(content_sequence1, style_batch)

                self.generator_valid(content_sequence1, content_sequence2, style_batch)

                self.logger.print_valid(e, self.epochs, vb, total_batch_valid)

            self.chkpt.step.assign_add(1)  # Save the current epochs number 

            self.training_evaluation(e)
            
            
            if e == 0:
                total_batch_train = i
                total_batch_valid = vb
        
        self.save()
        
    def multistyle_viz(self, epoch:int):
        
        for i, content_sequence in enumerate(self.content_viz_sequences):
        
            save_to = f"{self.logger.full_path}/{epoch}_{i}.png"  
        
            # Generate Sequences with the same content and all styles.
            
            content_sequences = tf.convert_to_tensor([content_sequence]* self.seed_styles_valid.shape[1])
            content_of_content = self.content_encoder(content_sequences)
            
            # Make the Style Space for the Real and Simulated Sequences.
            real_style_space = [ self.style_encoder(style_sequences) for style_sequences in self.seed_styles_valid ]
            
            # Generate sequence of different style, but with the same content.
            
            generated_sequences = np.array([ tf.concat(self.decoder([content_of_content, style_vectors]), -1) for style_vectors in real_style_space ])
            
            # Extract the content.
            content_of_gens = np.array([ self.content_encoder(generated_sequence) for generated_sequence in generated_sequences ])
                        
            
            # Extract the style.
            style_of_gen = np.array([ self.style_encoder(generated_sequence) for generated_sequence in generated_sequences ])
            
            # Reduce the dimentionality for the style.
            pca = PCA(2)
                    
            all_styles = tf.concat((real_style_space, style_of_gen), 0)
            all_styles = tf.reshape(all_styles, (-1, all_styles.shape[-1]))
                    
            pca = pca.fit(all_styles)
            
            real_reduced_styles = np.array([ pca.transform(particular_style_space) for particular_style_space in real_style_space ])
            gen_reduced_styles = np.array([ pca.transform(particular_style_space) for particular_style_space in style_of_gen ])
                        
            viz_content_of_content = np.expand_dims(content_of_content[0], 0) # Take only the output of the encoder of one sequence.
                
            visualization_helpers.plot_multistyle_sequences(
                content_sequence, 
                self.seed_styles_valid[:, 0], 
                generated_sequences[:, 0], 
                viz_content_of_content, content_of_gens[:, 0, :, :],
                real_reduced_styles, gen_reduced_styles,
                epoch, save_to
                )

    
    def training_evaluation(self, epoch):
        generation_style_train = np.array([self.generate(self.seed_content_train, style_train) for style_train in self.seed_styles_train])
        generation_style_valid = np.array([self.generate(self.seed_content_valid, style_train) for style_train in self.seed_styles_valid])

        mean_acc = self.classif_metric.evaluate(self.content_encoder, self.style_encoder, self.decoder)
        self.logger.valid_loggers["00 - Model Classification Acc"](mean_acc)
        
        if not ".npy" in self.task_arguments.content_dataset:
        
            metric_key = "00 - Proposed Metric"
            train_metric_values, valid_metric_values = [], []
            # Evaluate the metric for each style.
            for style_idx in range(self.seed_styles_train.shape[0]):
                
                real_style = self.seed_styles_train[style_idx]
                gen_style = generation_style_train[style_idx]
                metric_value = metric.compute_metric(gen_style, real_style, self.task_arguments)
                train_metric_values.append(metric_value)
                
                real_style = self.seed_styles_valid[style_idx]
                gen_style = generation_style_valid[style_idx]
                metric_value = metric.compute_metric(gen_style, real_style, self.task_arguments)
                valid_metric_values.append(metric_value)
                
            train_metric_values = np.mean(train_metric_values)
            valid_metric_values = np.mean(valid_metric_values)
            
            self.logger.train_loggers[metric_key](train_metric_values)
            self.logger.valid_loggers[metric_key](valid_metric_values)
        
            if valid_metric_values < self.best_metric_value:
                print(f"Curr metric: {valid_metric_values:0.2f}; best metric recorded: {self.best_metric_value}")
                print("Save the current model.")
                save_path = self.manager.save()
                print("Saved checkpoint for step {}: {}".format(int(self.chkpt.step), save_path))
                self.best_metric_value = valid_metric_values
                
        # exit()        
        self.simple_noise_metric(generation_style_train, generation_style_valid, epoch)

        self.simple_amplitude_metric(generation_style_train, generation_style_valid, epoch)
                
        # Make multistyle visualization. 
        self.multistyle_viz(epoch)
        
        self.logger.log_train(epoch)
        self.logger.log_valid(epoch)
        


            

    def simple_noise_metric(self, generation_style_train, generation_style_valid, epoch):
        
        placeholder = "01 - Noise Similarity Style"
        
        seed_content_trends_train, _ = simple_metric.simple_metric_on_noise(self.seed_content_train)
        seed_content_trends_valid, _ = simple_metric.simple_metric_on_noise(self.seed_content_valid)
        
        for i in range(generation_style_train.shape[0]):
            noise_key = f"{placeholder} {i+ 1}"
            content_key = f"02 - Content Similarity Style {i+ 1}"
            
            content_gen_train, generated_noise_train = simple_metric.simple_metric_on_noise(generation_style_train[i])
            content_gen_valid, generated_noise_valid = simple_metric.simple_metric_on_noise(generation_style_valid[i])
            
            _, seed_style_noise_train = simple_metric.simple_metric_on_noise(self.seed_styles_train[i])
            _, seed_style_noise_valid = simple_metric.simple_metric_on_noise(self.seed_styles_valid[i])
            
            noise_similarity_train = np.mean(np.abs(seed_style_noise_train - generated_noise_train))
            noise_similarity_valid = np.mean(np.abs(seed_style_noise_valid - generated_noise_valid))
            
            content_similarity_train = np.mean(np.abs(seed_content_trends_train - content_gen_train))
            content_similarity_valid = np.mean(np.abs(seed_content_trends_valid - content_gen_valid))
            
            self.logger.train_loggers[noise_key](noise_similarity_train)
            self.logger.valid_loggers[noise_key](noise_similarity_valid)
            
            self.logger.train_loggers[content_key](content_similarity_train)
            self.logger.valid_loggers[content_key](content_similarity_valid)
            
        train_mean_noise = self.logger.get_mean_metric(self.logger.train_loggers, 'Noise')
        valid_mean_noise = self.logger.get_mean_metric(self.logger.valid_loggers, 'Noise')
        
        self.logger.log_train_value(f"{placeholder} Mean", train_mean_noise, epoch)
        self.logger.log_valid_value(f"{placeholder} Mean", valid_mean_noise, epoch)
    

    def simple_amplitude_metric(self, generation_style_train, generation_style_valid, epoch):
        
        placeholder = f'03 - Amplitude Similarity Style'
        
        for i in range(generation_style_train.shape[0]):
            ampli_key = f"{placeholder} {i+ 1}"
            
            ampl_diff_train = simple_metric.simple_amplitude_metric(self.seed_styles_train[i], generation_style_train[i])
            ampl_diff_valid = simple_metric.simple_amplitude_metric(self.seed_styles_valid[i], generation_style_valid[i])

            self.logger.train_loggers[ampli_key](ampl_diff_train)
            self.logger.valid_loggers[ampli_key](ampl_diff_valid)
            
        train_mean_ampl = self.logger.get_mean_metric(self.logger.train_loggers, 'Amplitude')
        valid_mean_ampl = self.logger.get_mean_metric(self.logger.valid_loggers, 'Amplitude')
        
        self.logger.log_train_value(f'{placeholder} Mean', train_mean_ampl, epoch)
        self.logger.log_valid_value(f'{placeholder} Mean', valid_mean_ampl, epoch)

    def save(self):
        root = f"{self.shell_arguments.save_to}/{self.shell_arguments.exp_folder}/{self.shell_arguments.exp_name}"
        if not os.path.exists(root):
            print(f"[!] Save Folder Missing... Create root save folder at {root}")
            os.makedirs(root)

        print(f"[+] Saving to {root}")
        self.content_encoder.save(f"{root}/content_encoder.h5")
        self.style_encoder.save(f"{root}/style_encoder.h5")
        self.decoder.save(f"{root}/decoder.h5")
        self.global_discriminator.save(f"{root}/global_discriminator.h5")
        self.local_discriminator.save(f"{root}/local_discriminator.h5")
        print("[+] Save Parameters...")

        parameters = {
            "style_datasets":self.task_arguments.style_datasets, 
            "dset_content":self.task_arguments.content_dataset,
            "sequence_lenght_in_sample":self.task_arguments.sequence_lenght_in_sample,
            "granularity":self.task_arguments.granularity,
            "overlap":self.task_arguments.overlap,
            "epochs":self.epochs,
            "n_feature":self.task_arguments.n_feature,
            "n_classes":self.task_arguments.n_classes,
            "batch_size":self.task_arguments.batch_size,
            "style_vector_size":self.task_arguments.style_vector_size,
            "n_wiener":self.task_arguments.n_wiener,
            "n_sample_wiener":self.task_arguments.n_sample_wiener,
            "l_reconstr":self.task_arguments.l_reconstr,
            "l_local":self.task_arguments.l_local,
            "l_global":self.task_arguments.l_global,
            "l_style_preservation":self.task_arguments.l_style_preservation,
            "l_content":self.task_arguments.l_content,
            "encoder_adv":self.task_arguments.encoder_adv,
            "l_disentanglement":self.task_arguments.l_disentanglement,
            "l_triplet":self.task_arguments.l_triplet,
            "triplet_r":self.task_arguments.triplet_r,
            "task":self.shell_arguments.task,
            "univariate":self.task_arguments.univariate,
            "unsupervised":self.task_arguments.unsupervised
        }

        MLFlow_utils.save_configuration(f"{root}/model_config.json", parameters)
        print("[+] Saved !")

    def set_seeds(self, _seed_style_train, _seed_style_valid, content_viz_sequences):

        self.seed_styles_train = _seed_style_train
        self.seed_styles_valid = _seed_style_valid
        
        self.seed_content_train = get_batches(self.dset_content_train, 1)
        self.seed_content_valid = get_batches(self.dset_content_valid, 1)
        
        self.content_viz_sequences = content_viz_sequences


    def prepare_loggers(self, n_style:int):
        
        noise_metric = []
        ampli_metric = []
        content_sim_metric = []
        
        for i in range(n_style):
            noise_metric.append(f"01 - Noise Similarity Style {i+ 1}")    
            content_sim_metric.append(f"02 - Content Similarity Style {i+ 1}")
            ampli_metric.append(f"03 - Amplitude Similarity Style {i+ 1}")
            
        metric_keys = [
            "00 - Model Classification Acc",
            "00 - Proposed Metric",
            "10 - Total Generator Loss", 
            "11 - Reconstruction from Content",
            "111 - Local Reconstruction from Content",
            "12 - Central Realness",
            "13 - Local Realness",
            
            "20 - Style Loss",
            "21 - Triplet Loss",
            "22 - Disentanglement Loss",
            
            "30 - Content Loss",
            
            "40 - Global Discriminator Loss",
            "40 - Global Discriminator Acc", 
            
            "40 - Local Discriminator Loss",
            "40 - Local Discriminator Acc", 
            
            "41 - Global Discriminator Style Loss (Real Data)",
            "41 - Global Discriminator Style Loss (Fake Data)",
            
            "42 - Local Discriminator Style Loss (Real Data)",
            "42 - Local Discriminator Style Loss (Fake Data)",
            ]
        
        metric_keys.extend(noise_metric)
        metric_keys.extend(ampli_metric)
        metric_keys.extend(content_sim_metric)
        
        self.logger = TensorboardLog(self.log_folder, metric_keys)        

    def instanciate_datasets(self, 
                             content_dset_train:tf.data.Dataset, 
                             content_dset_valid:tf.data.Dataset, 
                             styles_dset_train:tf.data.Dataset, 
                             styles_dset_valid:tf.data.Dataset):
        
        self.dset_content_train = content_dset_train
        self.dset_content_valid = content_dset_valid

        self.dsets_style_train = styles_dset_train
        self.dsets_style_valid = styles_dset_valid


    @tf.function
    def discriminator_step(self, content_sequence1, style_batch):
        # Discriminator Step

        style_sequences, style_labels = style_batch[0], style_batch[1]
        
        with tf.GradientTape(persistent=True) as discr_tape:
            # Sequence generations.
            c1 = self.content_encoder(content_sequence1, training=False)

            style_encoded = self.style_encoder(style_sequences, training=False)
            
            generated= self.decoder([c1, style_encoded], training=False)
            
            # split sequences for discriminator's inputs
            real_style_sequences_splitted = tf.split(style_sequences, style_sequences.shape[-1], axis=-1)

            # Global on Real
            g_crit_real, g_style_classif_real = self.global_discriminator(real_style_sequences_splitted, training=True)

            # Local on Real
            l_crit_real = self.local_discriminator(real_style_sequences_splitted, training=True)

            # Global on Generated
            g_crit_fake, _ = self.global_discriminator(generated, training=True)

            # Local on fake
            l_crit_fake = self.local_discriminator(generated, training=True)
            
            # Compute the loss for GLOBAL the Discriminator
            # g_crit_loss = losses.discriminator_loss(g_crit_real, g_crit_fake)
            g_crit_loss = losses.least_square_discriminator_loss(g_crit_real, g_crit_fake)
            g_style_real = losses.style_classsification_loss(g_style_classif_real, style_labels)
            
            l_loss = losses.least_square_local_discriminator_loss(l_crit_real, l_crit_fake)

        # (GOBAL DISCRIMINATOR): Real / Fake and style
        global_discr_gradient = discr_tape.gradient([g_crit_loss, g_style_real], self.global_discriminator.trainable_variables)
        grads = discr_tape.gradient(l_loss, self.local_discriminator.trainable_variables)

        self.global_discriminator_opt.apply_gradients(zip(global_discr_gradient, self.global_discriminator.trainable_variables))  
        
        self.local_discriminator_opt.apply_gradients(zip(grads, self.local_discriminator.trainable_variables))
    

        # Calculate the performances of the Discriminator.
        real_labels = tf.ones_like(g_crit_real)
        generation_labels = tf.zeros_like(g_crit_fake)

        # Compute Accuracy for the GAN Training.
        # Thie accuracy will define if the generator has to be trained or the discriminator.

        # global_accs = tf.reduce_mean([
        #     binary_accuracy(real_labels, g_crit_real),
        #     binary_accuracy(generation_labels, g_crit_fake)
        # ])

        # channel_accs = tf.reduce_mean([
        #     losses.local_discriminator_accuracy(real_labels, l_crit_real),
        #     losses.local_discriminator_accuracy(generation_labels, l_crit_fake)
        # ])
        
        self.logger.train_loggers['40 - Global Discriminator Loss'](g_crit_loss)
        self.logger.train_loggers['41 - Global Discriminator Style Loss (Real Data)'](g_style_real)
        self.logger.train_loggers['40 - Local Discriminator Loss'](l_loss)

        # self.logger.train_loggers['40 - Global Discriminator Acc'](global_accs)
        # self.logger.train_loggers["40 - Local Discriminator Acc"](channel_accs)

    @tf.function
    def generator_step(self, content_sequence1, content_sequence2, style_batch):

        style_sequences, style_labels = style_batch[0],  style_batch[1]
        _bs = content_sequence1.shape[0]

        with tf.GradientTape() as content_tape, tf.GradientTape() as style_tape, tf.GradientTape() as decoder_tape:
            # Reconstruction Loss: Try to generate the same sequence given
            # it's content and style.
            contents = tf.concat([content_sequence1, content_sequence2], 0)
            cs = self.content_encoder(contents, training=True)
            s_cs = self.style_encoder(contents, training=True)
            id_generated = self.decoder([cs, s_cs], training=True)
            id_generated = tf.concat(id_generated, -1)

            global_reconstr = losses.recontruction_loss(contents, id_generated)
            local_reconstruction_loss = losses.local_recontruction_loss(contents, id_generated)
        
            ####
            styles = tf.concat([style_sequences, style_sequences], 0)   
            style_label_extended = tf.concat([style_labels, style_labels], 0)   
            
            encoded_content= self.content_encoder(contents, training=True)
            
            content_of_styles= self.content_encoder(styles, training=True)
            encoded_styles = self.style_encoder(styles, training=True)

            generations = self.decoder([encoded_content, encoded_styles], training=True)
            merged_generations = tf.concat(generations, -1)
            
            id_styles = self.decoder([content_of_styles, encoded_styles], training=True)
            id_styles = tf.concat(id_styles, -1 )
            
            global_style_reconstr = losses.recontruction_loss(styles, id_styles)
            local_style_recontruction = losses.local_recontruction_loss(styles, id_styles)
            
            global_reconstr = (global_reconstr + global_style_reconstr)/2
            local_reconstr = (local_reconstruction_loss + local_style_recontruction)/2
            

            s_generations = self.style_encoder(merged_generations, training=True)
            c_generations = self.content_encoder(merged_generations, training=True)

            # Discriminator pass for the adversarial loss for the generator.
            crit_on_fake, style_classif_fakes = self.global_discriminator(generations, training=False)

            # Local Discriminator on Fake Data.
            l_crit_on_fake = self.local_discriminator(generations, training=False)
            

            # Channel Generator losses (LSGAN)
            local_realness_loss = losses.least_square_local_generator_loss(l_crit_on_fake)
            
            # Global Generator losses.
            global_realness_loss = losses.least_square_generator_loss(crit_on_fake)
            # Style Classification loss
            global_style_loss = losses.style_classsification_loss(style_classif_fakes, style_label_extended)

            # Content encoder: Paths should be closes together.
            content_preservation = losses.fixed_point_content(encoded_content[-1], c_generations[-1])

            # Style encoder: Content should not influence the output of style encoder.
            s_c1_s = s_generations[:_bs]
            s_c2_s = s_generations[_bs:]
            content_style_disentenglement = losses.fixed_point_disentanglement(s_c2_s, s_c1_s, encoded_styles[:_bs])

            # Style encoder: Triplet loss.
            triplet_style1 = losses.hard_triplet(style_labels, s_c1_s, self.task_arguments.triplet_r)
            triplet_style2 = losses.hard_triplet(style_labels, s_c2_s, self.task_arguments.triplet_r)
            triplet_style = (triplet_style1+ triplet_style2)/2

    
            # content_encoder_loss = self.l_content* content_preservation+ self.e_adv* global_realness_loss + self.e_adv* global_style_loss + self.l_reconstr* reconstruction_part
            # style_encoder_loss = self.l_triplet* triplet_style + self.l_disentanglement* content_style_disentenglement  + self.e_adv* global_realness_loss + self.e_adv* global_style_loss + self.l_reconstr* reconstruction_part
            
            content_encoder_loss = self.l_content* content_preservation+ self.e_adv* global_realness_loss + self.encoder_recontr* global_reconstr
            style_encoder_loss = self.l_triplet* triplet_style + self.l_disentanglement* content_style_disentenglement  + self.e_adv* global_realness_loss + self.e_adv* global_style_loss + self.encoder_recontr* global_reconstr

            reconstruction_part = self.l_global_reconstr* global_reconstr+ self.l_local_reconstr* local_reconstr

            g_loss = self.l_reconstr* reconstruction_part + self.l_global* global_realness_loss + self.l_local* local_realness_loss + self.style_preservation* global_style_loss

        # Make the Networks Learn!
        content_grad=content_tape.gradient(content_encoder_loss, self.content_encoder.trainable_variables)
        style_grad = style_tape.gradient(style_encoder_loss, self.style_encoder.trainable_variables)
        decoder_grad = decoder_tape.gradient(g_loss, self.decoder.trainable_variables)
            
        self.opt_decoder.apply_gradients(zip(decoder_grad, self.decoder.trainable_variables))

        self.opt_content_encoder.apply_gradients(zip(content_grad, self.content_encoder.trainable_variables))
        self.opt_style_encoder.apply_gradients(zip(style_grad, self.style_encoder.trainable_variables))

        self.logger.train_loggers['10 - Total Generator Loss'](g_loss)
        self.logger.train_loggers["11 - Reconstruction from Content"](global_reconstr)
        self.logger.train_loggers["111 - Local Reconstruction from Content"](tf.reduce_mean(local_reconstr))

        self.logger.train_loggers["13 - Local Realness"](local_realness_loss)
        self.logger.train_loggers["12 - Central Realness"](global_realness_loss)

        self.logger.train_loggers["41 - Global Discriminator Style Loss (Fake Data)"](global_style_loss)

        self.logger.train_loggers["22 - Disentanglement Loss"](content_style_disentenglement)
        self.logger.train_loggers["21 - Triplet Loss"](triplet_style)
        self.logger.train_loggers["20 - Style Loss"](style_encoder_loss)
        self.logger.train_loggers["30 - Content Loss"](content_preservation)

    @tf.function
    def generator_valid(self, content_sequence1, content_sequence2, style_batch):
        
        style_sequences, style_labels = style_batch[0],  style_batch[1]
        
        # Reconstruction Loss: Try to generate the same sequence given
        # it's content and style.
        contents = tf.concat([content_sequence1, content_sequence2], 0)
        cs = self.content_encoder(contents, training=True)
        s_cs = self.style_encoder(contents, training=True)
        id_generated = self.decoder([cs, s_cs], training=True)
        id_generated = tf.concat(id_generated, -1)

        global_content_reconstr = losses.recontruction_loss(contents, id_generated)
        local_content_reconstr = losses.local_recontruction_loss(contents, id_generated)

        ####
        styles = tf.concat([style_sequences, style_sequences], 0)   
        style_label_extended = tf.concat([style_labels, style_labels], 0)   
        _bs = content_sequence1.shape[0]

        encoded_content= self.content_encoder(contents, training=True)
        content_of_styles= self.content_encoder(styles, training=True)
        encoded_styles = self.style_encoder(styles, training=True)

        generations = self.decoder([encoded_content, encoded_styles], training=True)
        merged_generations = tf.concat(generations, -1)
        
        id_styles = self.decoder([content_of_styles, encoded_styles], training=True)
        id_styles = tf.concat(id_styles, -1)
        
        global_style_reconstruction = losses.recontruction_loss(styles, id_styles)
        local_style_recontruction = losses.local_recontruction_loss(styles, id_styles)
        
        global_reconstr = (global_content_reconstr + global_style_reconstruction)/2
        local_reconstruction = (local_content_reconstr + local_style_recontruction)/2

        s_generations = self.style_encoder(merged_generations, training=True)
        c_generations = self.content_encoder(merged_generations, training=True)

        # Discriminator pass for the adversarial loss for the generator.
        crit_on_fake, style_classif_fakes = self.global_discriminator(generations, training=False)

        # Local Discriminator on Fake Data.
        l_crit_on_fake = self.local_discriminator(generations, training=False)

        # Channel Discriminator losses
        # local_realness_loss = losses.local_generator_loss(l_crit_on_fake)
        local_realness_loss = losses.least_square_local_generator_loss(l_crit_on_fake)
        
        # Global Generator losses.
        global_style_loss = losses.style_classsification_loss(style_classif_fakes, style_label_extended)
        # global_realness_loss = losses.generator_loss(crit_on_fake)
        global_realness_loss = losses.least_square_generator_loss(crit_on_fake)

        ########
        content_preservation = losses.fixed_point_content(encoded_content[-1], c_generations[-1])

        s_c1_s = s_generations[:_bs]
        s_c2_s = s_generations[_bs:]

        content_style_disentenglement = losses.fixed_point_disentanglement(s_c2_s, s_c1_s, encoded_styles[:_bs])

        triplet_style1 = losses.hard_triplet(style_labels, s_c1_s, self.task_arguments.triplet_r)
        triplet_style2 = losses.hard_triplet(style_labels, s_c2_s, self.task_arguments.triplet_r)
        triplet_style = (triplet_style1+ triplet_style2)/2

        # content_encoder_loss = self.l_content* content_preservation+ self.e_adv* global_realness_loss + self.e_adv* global_style_loss
        # style_encoder_loss = self.l_triplet* triplet_style + self.l_disentanglement* content_style_disentenglement  + self.e_adv* global_realness_loss + self.e_adv* global_style_loss

        content_encoder_loss = self.l_content* content_preservation+ self.e_adv* global_realness_loss + self.encoder_recontr* global_reconstr
        style_encoder_loss = self.l_triplet* triplet_style + self.l_disentanglement* content_style_disentenglement  + self.e_adv* global_realness_loss + self.e_adv* global_style_loss + self.encoder_recontr* global_reconstr  


        recontr_loss = self.l_global_reconstr* global_reconstr+ self.l_local_reconstr* local_reconstruction
        g_loss = self.l_reconstr* recontr_loss+ self.l_global* global_realness_loss + self.style_preservation* global_style_loss+ self.l_local* local_realness_loss

        self.logger.valid_loggers['10 - Total Generator Loss'](g_loss)
        self.logger.valid_loggers["11 - Reconstruction from Content"](global_reconstr)
        self.logger.valid_loggers["111 - Local Reconstruction from Content"](tf.reduce_mean(local_reconstruction))

        self.logger.valid_loggers["13 - Local Realness"](local_realness_loss)
        self.logger.valid_loggers["12 - Central Realness"](global_realness_loss)

        self.logger.valid_loggers["41 - Global Discriminator Style Loss (Fake Data)"](global_style_loss)

        self.logger.valid_loggers["21 - Triplet Loss"](triplet_style)
        self.logger.valid_loggers["20 - Style Loss"](style_encoder_loss)
        self.logger.valid_loggers["30 - Content Loss"](content_preservation)
        self.logger.valid_loggers["22 - Disentanglement Loss"](content_style_disentenglement)



    @tf.function
    def discriminator_valid(self, content_sequence1, style_batch):
        style_sequences, style_labels = style_batch[0], style_batch[1]

        # Sequence generations.
        c1 = self.content_encoder(content_sequence1, training=False)

        style_encoded = self.style_encoder(style_sequences, training=False)

        generated= self.decoder([c1, style_encoded], training=False)

        # split sequences for discriminator's inputs
        real_style_sequences_splitted = tf.split(style_sequences, style_sequences.shape[-1], axis=-1)

        # Global on Real
        g_crit_real, g_style_classif_real = self.global_discriminator(real_style_sequences_splitted, training=True)

        # Local on Real
        l_crit_real = self.local_discriminator(real_style_sequences_splitted, training=True)

        # Global on Generated
        g_crit_fake, _ = self.global_discriminator(generated, training=True)

        # Local on fake
        l_crit_fake = self.local_discriminator(generated, training=True)

        # Compute the loss for GLOBAL the Discriminator
        # g_crit_loss = losses.discriminator_loss(g_crit_real, g_crit_fake)
        g_crit_loss = losses.least_square_discriminator_loss(g_crit_real, g_crit_fake)
        g_style_real = losses.style_classsification_loss(g_style_classif_real, style_labels)
        
        # _real_output = tf.convert_to_tensor(l_crit_real)
        # _fake_output = tf.convert_to_tensor(l_crit_fake)
            
        # if len(_real_output.shape) < 3 and len(_fake_output.shape) < 3:
        #     l_crit_real = [l_crit_real]
        #     l_crit_fake = [l_crit_fake]

        # l_loss = losses.local_discriminator_loss(l_crit_real, l_crit_fake)
        l_loss = losses.least_square_local_discriminator_loss(l_crit_real, l_crit_fake)

        # Calculate the performances of the Discriminator.
        real_labels = tf.ones_like(g_crit_real)
        generation_labels = tf.zeros_like(g_crit_fake)

        # Compute Accuracy for the GAN Training.
        # Thie accuracy will define if the generator has to be trained or the discriminator.
        # global_accs = tf.reduce_mean([
        #     binary_accuracy(real_labels, g_crit_real),
        #     binary_accuracy(generation_labels, g_crit_fake)
        # ])

        # channel_accs = tf.reduce_mean([
        #     losses.local_discriminator_accuracy(real_labels, l_crit_real),
        #     losses.local_discriminator_accuracy(generation_labels, l_crit_fake)
        # ])
        
        self.logger.valid_loggers['40 - Global Discriminator Loss'](g_crit_loss)
        self.logger.valid_loggers['41 - Global Discriminator Style Loss (Real Data)'](g_style_real)
        self.logger.valid_loggers['40 - Local Discriminator Loss'](l_loss)

        # self.logger.valid_loggers['40 - Global Discriminator Acc'](global_accs)
        # self.logger.valid_loggers["40 - Local Discriminator Acc"](channel_accs)
    
