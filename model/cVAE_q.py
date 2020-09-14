import math
import os
import numpy as np
import seaborn as sns
import tensorflow as tf

import logging
import math
import threading
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from model.networks import encoder0, encoder, encoder_z, decoder, VAE_losses
# from utils.metrics import plot_co
from utils.pre_processing import risk_set, get_missing_mask, flatten_nested
from utils.tf_helpers import event_t_bin_prob, event_t_bin_prob_unif


class cVAE(object):
    def __init__(self,
                 batch_size,
                 learning_rate,
                 beta1,
                 beta2,
                 require_improvement,
                 seed,
                 num_iterations,
                 hidden_dim_encoder,
                 hidden_dim_decoder,
                 num_examples,
                 keep_prob,
                 train_data,
                 valid_data,
                 test_data,
                 covariates,
                 imputation_values,
                 sample_size,
                 categorical_indices,
                 max_epochs,
                 truncate_t = np.inf,
                 split_tt = 'percentile',
                 event_prob = 'empirical',
                 nbin = 100,
                 lambda_c = 1.0,
                 lambda_nc = 1.0,
                 output_dir=".",
                 path_large_data=""
                 ):
        

        self.truncate_t = truncate_t
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        # dimensions for encoders
        self.hidden_dim_encoder = hidden_dim_encoder
        # dimensions for the decoder
        self.hidden_dim_decoder = hidden_dim_decoder
        # dimensions of output discrete distribution
        self.nbin = nbin
        self.split_tt = split_tt
        self.event_prob = event_prob
        
        self.path_large_data = path_large_data
        self.seed = seed
        self.require_improvement = require_improvement
        self.num_iterations = num_iterations
        self.learning_rate, self.beta1, self.beta2 = learning_rate, beta1, beta2
        self.lambda_c, self.lambda_nc  = lambda_c, lambda_nc
        self.output_dir = os.path.dirname(output_dir)
        self.log_file = self.output_dir+'/model.log'
        logging.basicConfig(filename=self.log_file, filemode='w', level=logging.DEBUG)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.batch_norm = True
        self.covariates = covariates
        self.sample_size = sample_size

        self.config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        self.config.gpu_options.allow_growth = True
        # self.config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        # Load Data
        self.train_x, self.train_t, self.train_e = train_data['x'], train_data['t'], train_data['e']
        self.valid_x, self.valid_t, self.valid_e = valid_data['x'], valid_data['t'], valid_data['e']

        self.test_x, self.test_t, self.test_e = test_data['x'], test_data['t'], test_data['e']
        
        # calculate mean and variance for z-transformation
        self.norm_mean = np.nanmean(train_data['x'],axis=0)
        self.norm_std = np.nanstd(train_data['x'],axis=0)
        
        # split time t based on training set
        if self.split_tt == 'percentile':
            self.tt = np.percentile(train_data['t'][train_data['e']==1],np.linspace(0.,100.,self.nbin, endpoint=True))
        else:
            self.tt = np.linspace(np.floor(np.min(train_data['t'][train_data['e']==1])),np.ceil(np.max(train_data['t'][train['e']==1])),nbin, endpoint=True)
            
        if self.event_prob == 'empirical':
            event_tt_bin, self.event_tt_prob = event_t_bin_prob(train_data['t'], train_data['e'], self.tt)
        else:
            self.event_tt_prob = event_t_bin_prob_unif(self.tt)



#         self.end_t = end_t
        self.keep_prob = keep_prob
        self.input_dim = input_dim
        self.imputation_values = imputation_values
        self.num_examples = num_examples
        self.categorical_indices = categorical_indices
        self.continuous_indices = np.setdiff1d(np.arange(input_dim), flatten_nested(categorical_indices))
        print_features = "input_dim:{}, continuous:{}, size:{}, categorical:{}, " \
                         "size{}".format(self.input_dim,
                                         self.continuous_indices,
                                         len(
                                             self.continuous_indices),
                                         self.categorical_indices,
                                         len(
                                             self.categorical_indices))
        print(print_features)
        logging.debug(print_features)
        print_model = "model is cVAE "+"with "+self.split_tt " split of training time and "+ self.event_prob +' reweighting for censoring'
        print(print_model)
        logging.debug("Imputation values:{}".format(imputation_values))
        logging.debug(print_model)
        self.model = 'cVAE'

        self._build_graph()
        self.train_cost, self.train_ci, self.train_gen, self.train_disc,self.train_gan, self.train_t_reg, \
         = [], [], [], [], [],[]
        self.valid_cost, self.valid_ci, self.valid_gen, self.valid_disc, self.valid_gan, self.valid_t_reg \
        = [], [], [], [], [], []
        

    def _build_graph(self):
        self.G = tf.Graph()
        with self.G.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='x')
            self.event = tf.placeholder(tf.float32, shape=[None], name='event')
            self.t_ = tf.placeholder(tf.float32, shape=[None, self.nbin], name='t_')

            # are used to feed data into our queue
            self.batch_size_tensor = tf.placeholder(tf.int32, shape=[], name='batch_size')
            self.risk_set = tf.placeholder(tf.float32, shape=[None, None])
            self.impute_mask = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='impute_mask')
            self.is_training = tf.placeholder(tf.bool)

            self._objective()
            self.session = tf.Session(config=self.config)

            self.capacity = 1400
            self.coord = tf.train.Coordinator()
            enqueue_thread = threading.Thread(target=self.enqueue)
            self.queue = tf.RandomShuffleQueue(capacity=self.capacity, dtypes=[tf.float32, tf.float32, tf.float32],\
                                               shapes=[[self.input_dim], [self.input_dim], []],\
                                               min_after_dequeue=self.batch_size)
            # self.queue = tf.FIFOQueue(capacity=self.capacity, dtypes=[tf.float32, tf.float32, tf.float32],
            #                           shapes=[[self.input_dim], [], []])
            self.enqueue_op = self.queue.enqueue_many([self.x, self.t_, self.event])
            # enqueue_thread.isDaemon()
            enqueue_thread.start()
            dequeue_op = self.queue.dequeue()
            self.x_batch, self.t_batch, self.e_batch = tf.train.batch(dequeue_op, batch_size=self.batch_size,
                                                                      capacity=self.capacity)
            self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.session)

            self.saver = tf.train.Saver()
            self.merged = tf.summary.merge_all()
            self.current_dir = os.getcwd()
            # the / at the beginning was causing it to consider /tmp/my_model.ckpt the entire path, 
            # so it was saving the files in a directory called tmp directly on my hard drive. 
            # Removing the / put tmp in my ml folder.
            # self.save_path = self.output_dir+"/summaries/{0}_model".format(self.model)
            self.save_path = "summaries/{0}_model".format(self.model)
            self.train_writer = tf.summary.FileWriter(self.save_path, self.session.graph)
            self.save_model_path = self.output_dir+'/'+self.save_path+'.ckpt'

    def _objective(self):
        self.num_batches = self.num_examples / self.batch_size
        logging.debug("num batches:{}, batch_size:{} epochs:{}".format(self.num_batches, self.batch_size,
                                                                       int(self.num_iterations / self.num_batches)))
        self._build_model()

        self.cost = self.eloss + self.closs
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gradients = tf.gradients(loss, params)
        #gradients = tf.Print(gradients,[gradients], message ='gradients',summarize=2000)
        grads = zip(gradients, params) 
    
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.9, beta2=0.999)
        train_step = optimizer.apply_gradients(grads)
        
    def _build_model(self):
        self._risk_date()
    
    def _risk_date(self):
        # separate the input as event and censoring
        # we still keep observations in original order
        e_idx = tf.where(tf.equal(event, 1.))
        e_idx = tf.reshape(e_idx,[tf.shape(e_idx)[0]])
        ne_idx = tf.where(tf.equal(event, 0.))
        ne_idx = tf.reshape(ne_idx,[tf.shape(ne_idx)[0]])

        e_is_empty = tf.equal(tf.size(e_idx), 0)
        ne_is_empty = tf.equal(tf.size(ne_idx), 0)

        # Define VAE graph
        with tf.variable_scope('encoder0'):
            # update parameters encoder0 for all observations
            mu_logvar0 = encoder0(x, is_training)
            z0 = encoder_z(mu_logvar0)

        # update encoder q for both censoring and events
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            # with events, true t is t_;
            # for censoring, true time is t_r
            mu_logvar1 = encoder(x,t_, is_training)
            z1 = encoder_z(mu_logvar1)

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            # update for all samples
            t_logits_1 = decoder(z1, is_training)
            # update for all samples
            t_logits_0 = decoder(z0, is_training)

            # predict posterior distribution based on multiple z
            t_dist_new = tf.nn.softmax(t_logits_0)
            # Calculating average distribution
            t_dist_new_avg = t_dist_avg(mu_logvar0, t_dist_new, num_sample, is_training)

        # Optimization
        with tf.variable_scope('training') as scope:
            # calculate the losses separately, just for debugging purposes
            # calculate losses for events
            e_recon, e_kl_d, eloss = tf.cond(e_is_empty, lambda: zero_outputs(),\
                                             lambda:VAE_losses(tf.gather(t_logits_1,e_idx), tf.gather(t_,e_idx), \
                                 tf.gather(mu_logvar0,e_idx), tf.gather(mu_logvar1,e_idx)))

            # calculate losses for censor
            ne_recon, ne_kl_d, closs = tf.cond(ne_is_empty, lambda: zero_outputs(),\
                                               lambda: VAE_losses(tf.gather(t_logits_1,ne_idx), tf.gather(t_,ne_idx), \
                                 tf.gather(mu_logvar0,ne_idx), tf.gather(mu_logvar1,ne_idx)))        
        
       
        
        





        

