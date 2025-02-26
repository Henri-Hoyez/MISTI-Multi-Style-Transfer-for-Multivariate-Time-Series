import numpy as np
from configs.MetricConfig import Proposed
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import matplotlib as mpl

from utils.gpu_memory_grow import gpu_memory_grow

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf


gpus = tf.config.list_physical_devices('GPU')
gpu_memory_grow(gpus)



def draw_arrow(A, B, ax:plt.Axes, color="b", width=0.001):
    ax.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
              length_includes_head=True, color=color, width=width, alpha=0.95, head_width=width*4.0)
    
def draw_arrows(xs, ys, ax:plt.Axes, color="b", width=0.00025):
    # points = np.stack((xs, ys)).T
    # dist = points[1:] - points[:-1]

    # norms = np.linalg.norm(dist, axis=1)
    # norm = np.mean(norms)

    for i in range(xs.shape[0]-1):
        point0 = [xs[i], ys[i]]
        point1 = [xs[i+1], ys[i+1]]
        draw_arrow(point0, point1, ax, color=color, width=width)
        
        
def draw_content_space(
    ax:plt.Axes,
    content_wiener_process:tf.Tensor,
    color='tab:blue',
    label='An amayzing label.',
    arrow_width=0.00025):
    
    ax.scatter(content_wiener_process[:, 0], content_wiener_process[:, 1], label=label, color=color)
    draw_arrows(content_wiener_process[:, 0], content_wiener_process[:, 1], ax, color, width=arrow_width)


def plot_generated_sequence(
        content_encoder, style_encoder, decoder, 
        content_sequences, 
        seed_style_sequences:tf.Tensor, title="Default Title."):
    
    # seed style sequence:
    # Shape [n_style, nsequence, nfeat]

    # Generate viz for plots 
    print('[+] plotting')
    style_index = 0
    n_style = seed_style_sequences.shape[0]
    
    # cmap = mpl.colormaps['plasma']
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, n_style*2))
    
    c = np.array([content_sequences[0]])
    c = content_encoder(c)
    content_encoded = tf.concat([c]*n_style, 0)
    
    reshaped_style_sequences = tf.reshape(seed_style_sequences, (-1, seed_style_sequences.shape[-2], seed_style_sequences.shape[-1]))

    style_encoded = style_encoder(reshaped_style_sequences)
    
    pca = PCA(2)
    
    reduced_style = pca.fit_transform(style_encoded)
    
    reduced_style = tf.reshape(reduced_style, (
        seed_style_sequences.shape[0],
        style_encoded.shape[0]//seed_style_sequences.shape[0], 
        -1))
    
    reshaped_style_encoded = tf.reshape(style_encoded, (
        seed_style_sequences.shape[0],
        style_encoded.shape[0]//seed_style_sequences.shape[0], 
        -1))

    generated_viz = decoder([content_encoded, reshaped_style_encoded[:, style_index, :]])
    
    generated_viz = tf.concat(generated_viz, -1)

    all_values = np.array([
        content_sequences[0], 
        seed_style_sequences[0, style_index], 
        seed_style_sequences[1, style_index]
        ])
    
    _min, _max = np.min(all_values)-1, np.max(all_values)+ 1
    
    content_of_generated_viz = content_encoder(generated_viz)

    content_of_style_sequences = content_encoder(seed_style_sequences[:, style_index, :])
    
    # Make point for Style Scatter plot. 
    c1s = tf.convert_to_tensor([c[0]]* reshaped_style_sequences.shape[0])
    
    with tf.device("cpu:0"):
        gen_c1_s = decoder([c1s, style_encoded])
    gen_c1_s = tf.concat(gen_c1_s, -1)
    
    style_of_generated_viz = style_encoder(gen_c1_s)
    
    reduced_style_of_generated_viz = pca.transform(style_of_generated_viz)
        
    reduced_style_of_generated_viz = tf.reshape(reduced_style_of_generated_viz, (
        seed_style_sequences.shape[0],
        style_encoded.shape[0]//seed_style_sequences.shape[0], 
        -1))
    
    cacateneted = tf.concat((content_of_generated_viz, content_of_style_sequences), 0)
    
    x_min, x_max = np.min(cacateneted[:, 0]),  np.max(cacateneted[:, 0])
    y_min, y_max =  np.min(cacateneted[:, 1]),  np.max(cacateneted[:, 1])
    diag = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
    
    fig = plt.figure(figsize=(25, 15))
    fig.suptitle(title, fontsize=25)
    spec= fig.add_gridspec(3, 8)
    
    
    ax00 = fig.add_subplot(spec[0:2, :2])
    ax00.set_title('Content Sequence. ($C_0$)')
    ax00.plot(content_sequences[0])
    ax00.set_ylim(_min, _max)
    ax00.grid(True)
    ax00.legend()

# #######
    ax01 = fig.add_subplot(spec[0, 2:5])
    ax01.set_title('Style Sequence 1. ($S_0$)')
    ax01.plot(seed_style_sequences[0, style_index, :])
    ax01.set_ylim(_min, _max)

    ax01.grid(True)

    ax11 = fig.add_subplot(spec[1, 2:5])
    ax11.set_title("Style Sequence 2. ($S_1$)")
    ax11.plot(seed_style_sequences[1, style_index, :])
    ax11.set_ylim(_min, _max)
    ax11.grid(True)

# #######
    ax02 = fig.add_subplot(spec[0, 5:])
    ax02.set_title('Generated Sequence. ($C_0; S_0$)')
    ax02.plot(generated_viz[0])
    ax02.set_ylim(_min, _max)
    ax02.grid(True) 

    ax12 = fig.add_subplot(spec[1, 5:])
    ax12.set_title('Generated Sequence. ($C_0; S_1$)')
    ax12.plot(generated_viz[1])
    ax12.set_ylim(_min, _max)
    ax12.grid(True) 
    
    
    ax10 = fig.add_subplot(spec[2, :4])
    ax10.set_title('Content Space.'+ f"{diag}")
    
    ax11 = fig.add_subplot(spec[2, 4:])
    ax11.set_title('Style Space, Reduced with PCA.')   
     
    for i in range(n_style):
        colors[2*i+1]
        draw_content_space(ax10, content_of_style_sequences[i][-1], color=colors[2*i], label=f"content space Real style {i+1}", arrow_width=.0005*diag)
        draw_content_space(ax10, content_of_generated_viz[i][-1], color=colors[2*i+1], label=f"content space Gen  style {i+1}", arrow_width=.0005*diag)
        
        ax11.scatter(
            reduced_style[i, :150, 0], 
            reduced_style[i,:150, 1], 
            label=f'Real Style {i}.', 
            alpha=0.25, 
            color=colors[2*i]
            )
        
        ax11.scatter(
            reduced_style_of_generated_viz[i, :150, 0], 
            reduced_style_of_generated_viz[i,:150, 1], 
            label=f'Generated Style {i}.', alpha=0.25,
            color=colors[2*i+1]
            )
        
        
    ax10.legend()
    ax11.legend()
    
    ax10.grid()    
    ax11.grid()
    
    plt.tight_layout()
    
    return fig


######### 

def make_graph_on_subplot(ax:plt.Axes, sequence:tf.Tensor, title:str):
    ax.set_title(title)
    
    ax.plot(sequence)
    ax.grid(True)
    
    return ax


def plot_multistyle_sequences(
    content_sequence:tf.Tensor, 
    style_sequences:tf.Tensor, 
    generated_sequence:tf.Tensor, 
    real_content_space, gen_content_space, 
    real_style_space, gen_style_space,
    epoch:int, to_file:str=None):
    
    n_style = style_sequences.shape[0]    
    
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, n_style*2))
    
    all_values = np.concatenate((np.expand_dims(content_sequence, axis=0), style_sequences), axis=0)
    _min, _max = np.min(all_values)-0.5, np.max(all_values)+ 0.5
    
    print(real_content_space.shape, gen_content_space.shape)
    
    cacateneted = tf.concat((real_content_space, gen_content_space), 0)
    
    x_min, x_max = np.min(cacateneted[:, 0]),  np.max(cacateneted[:, 0])
    y_min, y_max =  np.min(cacateneted[:, 1]),  np.max(cacateneted[:, 1])
    diag = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
        
    fig = plt.figure(figsize=(25, 30))
    fig.suptitle(f"Multiple Style Generation Epoch: {epoch}", fontsize=25)
    spec= fig.add_gridspec(n_style+ 2, 2)
    
    ax00 = fig.add_subplot(spec[0, :])
    
    make_graph_on_subplot(ax00, content_sequence, "Content Sequence.")
    ax00.set_ylim(_min, _max)
    
    ax_content = fig.add_subplot(spec[-1, 0])
    ax_content.set_title("Content Space.")
    
    
    draw_content_space(ax_content, real_content_space[0], color=colors[0], label=f"Content Sequence.", arrow_width=.0005*diag) 
    
    ax_content.grid(True)
    
    ax_style = fig.add_subplot(spec[-1, 1])
    ax_style.set_title("Style Space.")
    ax_style.grid(True)
    
    for i in range(n_style):
        axi1 = fig.add_subplot(spec[i+ 1, 0])
        axi1 = make_graph_on_subplot(axi1, style_sequences[i], f"Style {i+ 1}.")
        axi1.set_ylim(_min, _max)
        
        axi2 = fig.add_subplot(spec[i+ 1, 1])
        axi2 = make_graph_on_subplot(axi2, generated_sequence[i], f"Generated sequence on Style {i+ 1}.")
        axi2.set_ylim(_min, _max)
        
        ax_style.scatter(
            real_style_space[i, :150, 0], 
            real_style_space[i,:150, 1], 
            label=f'Real Style {i}.', 
            alpha=0.25, 
            color=colors[2*i]
            )
        
        ax_style.scatter(
            gen_style_space[i, :150, 0], 
            gen_style_space[i,:150, 1], 
            label=f'Gen Style {i}.', 
            alpha=0.25, 
            color=colors[2*i+ 1]
            )
        
        draw_content_space(ax_content, gen_content_space[i], color=colors[2*i+1], label=f"Content of Gen Style {i+1}", arrow_width=.0005*diag) 
    
    
    ax_style.legend(bbox_to_anchor=(1.0, 1.038), loc='upper left')
    ax_content.legend(bbox_to_anchor=(-.1, 1.038), loc='upper right')
    plt.tight_layout()
    
    if not to_file is None:
        plt.savefig(to_file)
    