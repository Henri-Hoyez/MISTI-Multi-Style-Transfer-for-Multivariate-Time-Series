import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

import umap

from models.evaluation.eval_classifiers import make_content_space_classif
from models.NaiveClassifier import make_naive_discriminator


def make_embedding(model:tf.keras.models.Model, dataset:np.ndarray):
    _out = model.layers[-3].output
        
    embeder = tf.keras.Model(model.input, _out)
            
    embeddings = np.array([embeder(d[0]) for d in dataset])
    shape = embeddings.shape
    
    embeddings = embeddings.reshape((-1, shape[-1]))
    
    return np.array(embeddings)

def get_f1(precision, recall):
    return np.mean(2 * (precision * recall)/(precision + recall))


def train_naive_discriminator(train_dset, valid_dset, training_params:dict, epochs, n_classes):
    seq_shape = (training_params["sequence_lenght_in_sample"], training_params["n_feature"])

    naive_discr = make_naive_discriminator(seq_shape, n_classes)
    history = naive_discr.fit(train_dset, validation_data=valid_dset, epochs=epochs, verbose=1)

    return naive_discr.evaluate(valid_dset)[1], history, naive_discr


def umap_plot(real_style1, real_style2, gen_style1, gen_style2, root_folder, title_extenssion=""):
    save_to= f"{root_folder}/UMAP_plot.png"
    n_sequences= real_style1.shape[0]

    concatenated = tf.concat((real_style1, real_style2, gen_style1, gen_style2), 0)

    concatenated = tf.transpose(concatenated, (0,2, 1))
    concatenated = tf.reshape(concatenated, (concatenated.shape[0], -1))

    _mean, _std = tf.math.reduce_mean(concatenated), tf.math.reduce_std(concatenated)

    concatenated = (concatenated - _mean)/_std

    reducer = umap.UMAP(n_neighbors=150, min_dist=1.)
    reduced = reducer.fit_transform(concatenated)

    style1_reduced = reduced[:n_sequences]
    style2_reduced = reduced[n_sequences:2*n_sequences]
    generated_s1_reduced = reduced[2*n_sequences:3*n_sequences]
    generated_s2_reduced = reduced[3*n_sequences:]

    plt.figure(figsize=(18, 10))
    plt.scatter(style1_reduced[:, 0], style1_reduced[:, 1], label="Real Style 1", alpha=0.25)
    plt.scatter(style2_reduced[:, 0], style2_reduced[:, 1], label="Real Style 2", alpha=0.25)
    plt.scatter(generated_s1_reduced[:, 0], generated_s1_reduced[:, 1], label="Generated Style 1", alpha=0.25)
    plt.scatter(generated_s2_reduced[:, 0], generated_s2_reduced[:, 1], label="Generated Style 2", alpha=0.25)

    plt.grid()
    plt.title(f"UMAP Reduction on Time Series. {title_extenssion}", fontsize=15)
    plt.ylabel("y_umap", fontsize=15)
    plt.xlabel("x_umap", fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig(save_to, bbox_inches='tight')
    plt.show()



def tsne_plot(real_style1, real_style2, gen_style1, gen_style2, root_folder, title_extenssion=""):
    save_to= f"{root_folder}/t-SNE_plot.png"
    n_sequences= real_style1.shape[0]

    concatenated = tf.concat((real_style1, real_style2, gen_style1, gen_style2), 0)

    concatenated = tf.transpose(concatenated, (0,2, 1))
    concatenated = tf.reshape(concatenated, (concatenated.shape[0], -1))

    _mean, _std = tf.math.reduce_mean(concatenated), tf.math.reduce_std(concatenated)

    concatenated = (concatenated - _mean)/_std

    reducer = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=150)
    reduced = reducer.fit_transform(concatenated)

    style1_reduced = reduced[:n_sequences]
    style2_reduced = reduced[n_sequences:2*n_sequences]
    generated_s1_reduced = reduced[2*n_sequences:3*n_sequences]
    generated_s2_reduced = reduced[3*n_sequences:]

    plt.figure(figsize=(18, 10))
    plt.scatter(style1_reduced[:, 0], style1_reduced[:, 1], label="Real Style 1", alpha=0.25)
    plt.scatter(style2_reduced[:, 0], style2_reduced[:, 1], label="Real Style 1", alpha=0.25)
    plt.scatter(generated_s1_reduced[:, 0], generated_s1_reduced[:, 1], label="Generated Style 1", alpha=0.25)
    plt.scatter(generated_s2_reduced[:, 0], generated_s2_reduced[:, 1], label="Generated Style 2", alpha=0.25)

    plt.grid()
    plt.title(f"t-SNE Reduction on Time Series. {title_extenssion}", fontsize=15)
    plt.ylabel("y_tsne", fontsize=15)
    plt.xlabel("x_tsne", fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig(save_to, bbox_inches='tight')
    plt.show()


def predictions_on_content_space(train_dset:tf.data.Dataset, valid_dset:tf.data.Dataset, args:dict):
    n_sample_wiener = args.simulated_arguments.n_sample_wiener
    n_feature_wiener= args.simulated_arguments.n_wiener
    n_labels = 5
    epochs = 1
    
    classifier = make_content_space_classif(n_sample_wiener, n_feature_wiener, n_labels)

    history = classifier.fit(train_dset, validation_data=valid_dset, epochs=epochs)

    # plt.figure(figsize=(18, 10))
    # plt.plot(history.history["loss"], label="loss")
    # plt.plot(history.history["val_loss"], label="val loss")
    # plt.grid()
    # plt.legend()
    # plt.savefig("allo.png")

    return classifier.evaluate(valid_dset)[1]




