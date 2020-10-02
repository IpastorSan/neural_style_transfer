#3 different implementations of Gram matrix.
import numpy as np
import tensorflow as tf

def gram_matrix_1(x):
    # input is (1, Height, Width, C) ---> squeeze to (H,W,C)
    x = tf.squeeze(x)
    x = tf.keras.backend.batch_flatten(tf.keras.backend.permute_dimensions(x, pattern=(2, 0, 1)))  # shape (C, H*W)
    # average this outer product across all locations
    num_locations = tf.cast(tf.shape(x)[1], tf.float32)
    gram = tf.keras.backend.dot(x, tf.keras.backend.transpose(x)) / num_locations
    return gram

def gram_matrix_2(x):
    #input is (1, Height, Width, C) ---> squeeze to (H,W,C)
    x = tf.squeeze(x)
    x = tf.transpose(x, (2,0,1))   #shape (C,H,W)
    features = tf.reshape(x, (tf.shape(x)[0], -1))  #shape (C, H*W)
    gram = tf.matmul(features, tf.transpose(features)) #outer product of features and its transposed
    return gram

def gram_matrix_3(x):
    # input is (1, Height, Width, C) ---> squeeze to (H,W,C)
    #Take the outer product of the feature vector with itself at each location, then average this outer product accross
    #all locations
    result = tf.linalg.einsum("bijc, bijd->bcd", x, x) #batch matrix multiplication, shape (1, 3, 3)
    input_shape = tf.shape(x)
    num_locations = tf.cast(input_shape[1]* input_shape[2], tf.float32) #(Heigh, Width)
    return result/num_locations
