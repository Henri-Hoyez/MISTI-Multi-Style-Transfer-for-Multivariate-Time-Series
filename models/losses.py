import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf
import numpy as np

from tensorflow.python.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.python.keras.metrics import binary_accuracy


cross_entropy = BinaryCrossentropy(from_logits=False)
error_classif = SparseCategoricalCrossentropy()


@tf.function
def recontruction_loss(true:tf.Tensor, generated:tf.Tensor):
    
    diff = tf.math.square(true- generated)
    # l2s = tf.math.reduce_sum(diff, axis=0)
    
    result = tf.reduce_mean(diff)
        
    return tf.convert_to_tensor([result])

@tf.function
def local_recontruction_loss(true:tf.Tensor, generated:tf.Tensor):
    
    diff = tf.math.square(true- generated)
    # l2s = tf.math.reduce_sum(diff, axis=1)

    results = tf.reduce_mean(diff, axis=(1, 0))
     
    return results
     
@tf.function
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

@tf.function
def generator_loss(fake_output):
    return tf.convert_to_tensor([cross_entropy(tf.ones_like(fake_output), fake_output)])

@tf.function
def fixed_point_content(encoded_content_real, encoded_content_fake):
    diff = encoded_content_fake- encoded_content_real
    # return tf.reduce_mean(tf.square(diff))
    return tf.reduce_sum(tf.square(diff))

@tf.function
def style_classsification_loss(y_pred, y_true):
    return tf.convert_to_tensor([error_classif(y_true, y_pred)])

@tf.function
def local_generator_loss(crit_on_fakes:list):
    individual_losses = []

    for crit_on_fake in crit_on_fakes:
        individual_losses.append(cross_entropy(tf.ones_like(crit_on_fake), crit_on_fake))
        
    return tf.convert_to_tensor(individual_losses)

@tf.function
def local_discriminator_loss(crits_on_real, crits_on_fake):
    individual_losses = []

    for local_real, local_fake in zip(crits_on_real, crits_on_fake):
        l1 = cross_entropy(tf.ones_like(local_real), local_real)
        l2 = cross_entropy(tf.zeros_like(local_fake), local_fake)
        loss = l1+ l2
        individual_losses.append(loss)
        
    return individual_losses

@tf.function
def local_discriminator_accuracy(y_true, y_preds):
    # Y_true [BS, 1]
    #y_preds: [n_signals, BS, 1]

    accs = []
    for y_pred in y_preds:
        accs.append(binary_accuracy(y_true, y_pred))

    return tf.reduce_mean(accs)


# ###                 ### #
# LEAST SQUARE GAN LOSSES #
# ###                 ### # 
@tf.function
def least_square_discriminator_loss(real_output, fake_output):
    a = 1
    b = 0
    
    _real_gt = tf.zeros_like(real_output) + a
    _fake_gt = tf.zeros_like(fake_output) + b
    
    loss = 0.5* tf.reduce_mean(tf.square(real_output - _real_gt)) + 0.5* tf.reduce_mean(tf.square(fake_output - _fake_gt))
    return loss

@tf.function
def least_square_generator_loss(fake_output):
    c = 1 
    _fake_gt = tf.zeros_like(fake_output) + c
    
    loss = 0.5* tf.reduce_mean(tf.square(fake_output - _fake_gt))
    return loss

@tf.function
def least_square_local_generator_loss(fake_output):
    individual_losses = []

    for crit_on_fake in fake_output:
        individual_losses.append(least_square_generator_loss(crit_on_fake))
        
    return tf.convert_to_tensor(individual_losses)

@tf.function
def least_square_local_discriminator_loss(real_output, fake_output):
    individual_losses = []

    for local_real, local_fake in zip(real_output, fake_output):
        _loss = least_square_discriminator_loss(local_real, local_fake)
        individual_losses.append(_loss)
        
    return individual_losses



# ###
@tf.function
def l2(x:tf.Tensor, y:tf.Tensor):
    diff = tf.square(y- x)
    _distance = tf.reduce_sum(diff, axis=-1)
    return _distance

def path_area(content_path:tf.Tensor):
    # We will suppose that the encoded path is bahaving inside a 
    # Rectangle
     
    _x_min = tf.reduce_min(content_path[:, :, 0], axis=-1)
    _x_max = tf.reduce_max(content_path[:, :, 0], axis=-1)
    
    _y_min = tf.reduce_min(content_path[:, :, 1], axis=-1)
    _y_max = tf.reduce_max(content_path[:, :, 1], axis=-1)
    
    side1 = tf.abs(_x_max - _x_min)
    side2 = tf.abs(_y_max - _y_min)
    
    return side1 * side2

@tf.function
def mean_content_distance(content_path: tf.Tensor):
    
    diff = tf.square(content_path[:, 1:]- content_path[:, :-1])
    diff = tf.sqrt(tf.reduce_sum(diff, axis=-1))

    return tf.reduce_mean(diff)
    
    
@tf.function
def fixed_point_content(encoded_content_real, encoded_content_fake):
    diff = l2(encoded_content_real, encoded_content_fake)
    
    mean_distance = mean_content_distance(encoded_content_real)
    
    return tf.reduce_mean(diff)/mean_distance

@tf.function
def _pairwise_distance(a_embeddings, b_embeddings):
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(a_embeddings, tf.transpose(b_embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    return distances

@tf.function
def select_index(pos_list:tf.Tensor):
    pos_indexes = tf.where(pos_list == 1)
    return tf.random.shuffle(tf.reshape(pos_indexes, (-1,)))[0]

@tf.function
def select_positive_negative(labels:tf.Tensor, embeddings:tf.Tensor):
    # Get a Square matrix, where there is ones when the class is the same.
    positives = tf.equal(labels, tf.transpose(labels))
    negatives = tf.logical_not(positives)

    positives = tf.cast(positives, tf.float32)
    negatives = tf.cast(negatives, tf.float32)
    
    positives = positives - tf.eye(labels.shape[0])

    pos_embs, negs_embs = [], []
    for i in range(positives.shape[0]):
        pos, neg = positives[i], negatives[i]

        pos_index = select_index(pos)
        neg_index = select_index(neg)

        pos_embs.append(embeddings[pos_index])
        negs_embs.append(embeddings[neg_index])

    return tf.convert_to_tensor(embeddings), tf.convert_to_tensor(pos_embs), tf.convert_to_tensor(negs_embs)

@tf.function
def get_triplet_loss(anchor_embedding, positive_embedding, negative_embedding, triplet_r=0.5):
    positive_distance= _pairwise_distance(anchor_embedding, positive_embedding)
    negative_distance= _pairwise_distance(anchor_embedding, negative_embedding)

    positive_index= tf.argmax(positive_distance, axis=1)

    pos_embedding = tf.gather(positive_embedding, positive_index)
 
    neg_indexes = tf.argmin(negative_distance, axis=1)
    
    neg_embeddings= tf.gather(negative_embedding, neg_indexes)

    positive_distances= l2(anchor_embedding, pos_embedding)
    negative_distances= l2(anchor_embedding, neg_embeddings)

    loss = tf.reduce_mean(tf.maximum(triplet_r+ positive_distances - negative_distances, 0))

    return loss
@tf.function
def hard_triplet(labels:tf.Tensor, embeddings:tf.Tensor, triplet_r=0.5):

    distances = _pairwise_distance(embeddings, embeddings)

    # Get a Square matrix, where there is ones when the class is the same.
    positives = tf.equal(labels, tf.transpose(labels))
    negatives = tf.logical_not(positives)

    positives = tf.cast(positives, tf.float32)
    negatives = tf.cast(negatives, tf.float32)
    
    positives = positives - tf.eye(labels.shape[0])

    positive_distances = distances* positives
    negatives_distances = distances* negatives
     
    negatives_distances = tf.where(tf.equal(negatives_distances, 0.), np.inf, negatives_distances)

    positive_index= tf.argmax(positive_distances, axis=1)
    pos_embs = tf.gather(embeddings, positive_index)

    negative_index = tf.argmin(negatives_distances, axis=1)
    neg_embs =  tf.gather(embeddings, negative_index)

    positive_distances= l2(embeddings, pos_embs)
    negative_distances= l2(embeddings, neg_embs)

    loss = tf.reduce_mean(tf.maximum(triplet_r+ positive_distances - negative_distances, 0))

    return loss

@tf.function
def fixed_point_disentanglement(
        es_x1_y:tf.Tensor, 
        es_x2_y:tf.Tensor, 
        es_y:tf.Tensor
        ):

    diff1 = l2(es_x1_y, es_x2_y)
    diff2 = l2(es_x1_y, es_y)

    loss = diff1- diff2
    zeros = tf.zeros_like(loss)
    loss = tf.math.maximum(loss, zeros)
    loss = tf.reduce_mean(loss)
    return loss


