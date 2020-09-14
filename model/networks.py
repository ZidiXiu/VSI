import math
import os
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim
# start with 3 layers each
def encoder0(x,is_training):
    """learned prior: Network p(z|x)"""
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
#                         normalizer_fn=slim.batch_norm,
#                         normalizer_params={'is_training': is_training},
                    weights_initializer=tf.contrib.layers.xavier_initializer()):

        mu_logvar = slim.fully_connected(x, 32, scope='fc1')
        mu_logvar = slim.fully_connected(mu_logvar, 32, scope='fc2')
        mu_logvar = slim.fully_connected(mu_logvar, 32, activation_fn=None, scope='fc3')
        
    return mu_logvar


def encoder(x,t_, is_training):
    """Network q(z|x,t_)"""
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
#                         normalizer_fn=slim.batch_norm,
#                         normalizer_params={'is_training': is_training},
                    weights_initializer=tf.contrib.layers.xavier_initializer()):
        inputs = tf.concat([t_,x],axis=1)
        mu_logvar = slim.fully_connected(inputs, 32, scope='fc1')
        mu_logvar = slim.fully_connected(mu_logvar, 32, scope='fc2')
        mu_logvar = slim.fully_connected(mu_logvar, 32, activation_fn=None, scope='fc3')
        
    return mu_logvar

def encoder_z(mu_logvar, epsilon=None):
        
    # Interpret z as concatenation of mean and log variance
    mu, logvar = tf.split(mu_logvar, num_or_size_splits=2, axis=1)

    # Standard deviation must be positive
    stddev = tf.sqrt(tf.exp(logvar))
    
    if epsilon is None:
        # Draw a z from the distribution
        epsilon = tf.random_normal(tf.shape(stddev))
        
    z = mu + tf.multiply(stddev, epsilon)        
        
    return z


def decoder(z, is_training):
    """Network p(t|z)"""
    # Decoding arm
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
#                         normalizer_fn=slim.batch_norm,
#                         normalizer_params={'is_training': is_training},
                    weights_initializer=tf.contrib.layers.xavier_initializer()):
        t_logits = slim.fully_connected(z, 32, scope='fc1')
        t_logits = slim.fully_connected(t_logits, 32, scope='fc2')
        t_logits = slim.fully_connected(t_logits, 32, scope='fc3')
       
        # returns multinomial distribution
        t_logits = slim.fully_connected(t_logits, nbin, activation_fn=None, scope='fc4')
        # t_logits = tf.nn.softmax(t_logits)
        
    return (t_logits)

def VAE_losses(t_logits, t_, mu_logvar0, mu_logvar1, tiny=1e-8):
    """Define loss functions (reconstruction, KL divergence) and optimizer"""
    # Reconstruction loss
    reconstruction = tf.nn.softmax_cross_entropy_with_logits_v2(labels=t_, logits=t_logits)

    # KL divergence
    mu0, logvar0 = tf.split(mu_logvar0, num_or_size_splits=2, axis=1)
    mu1, logvar1 = tf.split(mu_logvar1, num_or_size_splits=2, axis=1)

    kl_d = 0.5 * tf.reduce_sum(tf.divide(tf.exp(logvar1),tf.exp(logvar0)+tiny)\
                               + tf.divide(tf.square(mu0-mu1),tf.exp(logvar0)+tiny) \
                               + logvar0 - logvar1 -1.0, \
                               1)

    # Total loss for event
    loss = tf.reduce_mean(reconstruction + kl_d)        
    
    return reconstruction, kl_d, loss

