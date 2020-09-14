import math
import os
import sys
import pandas

import numpy as np
import seaborn as sns
import tensorflow as tf
import logging

# from utils.simulation_functions import simulation_cox_gompertz
from utils.preprocessing import formatted_data, normalize_batch, event_t_bin_prob,risk_t_bin_prob,\
batch_t_categorize, next_batch, one_hot_encoder, one_hot_indices, flatten_nested
from utils.metrics import calculate_quantiles, random_multinomial, MVNloglikeli_np, random_uniform_p
# simulation settings

name = 'cVAE_q_flchain'
### on my mac
# directory of output model
# output_dir = '/Users/ZidiXiu/Dropbox/Research/VAE/datasets/flchain'+'/'
# directory of output test results
# out_dir = '/Users/ZidiXiu/Dropbox/Research/VAE/results/flchain'+'/'
# flchain dataset
# file_path = '/Users/ZidiXiu/Dropbox/Research/VAE/datasets'

### on GPU server
# directory of output model
output_dir = '/data/zidi/cVAE/results/flchain/saved_models'+'/'
log_file = output_dir+name+'.log'
logging.basicConfig(filename=log_file, filemode='w', level=logging.DEBUG)
# directory of output test results
out_dir = '/data/zidi/cVAE/results/flchain'+'/'
# flchain dataset
file_path = '/data/zidi/cVAE/datasets/'



training = True

path = os.path.abspath(os.path.join(file_path, '', 'flchain.csv'))
data_frame = pandas.read_csv(path, index_col=0)
# remove rows with 0 time-to-event
data_frame = data_frame[data_frame.futime != 0]
data_frame['pat'] = np.arange(data_frame.shape[0])
# x_data = data_frame[['age', 'sex', 'kappa', 'lambda', 'flc.grp', 'creatinine', 'mgus']]
# Preprocess

to_drop = ['futime', 'death', 'chapter', 'pat']
dataset = data_frame.drop(labels=to_drop, axis=1)

one_hot_encoder_list = ['sex', 'flc.grp', 'sample.yr']
one_hot_encoder_list_idx = np.where(np.isin(dataset.columns.values, np.array(one_hot_encoder_list)))

# split to train/valid/test before calculating imputation values
# first shuffling all indices
idx = np.arange(0, dataset.shape[0])
np.random.seed(123)
np.random.shuffle(idx)
num_examples = int(0.80 * dataset.shape[0])
print("num_examples:{}".format(num_examples))
train_idx = idx[0: num_examples]
split = int(( dataset.shape[0] - num_examples) / 2)
test_idx = idx[num_examples: num_examples + split]
valid_idx = idx[num_examples + split:  dataset.shape[0]]

####
t_data = data_frame[['futime']]
e_data = data_frame[['death']]
pat_data = data_frame[['pat']]

# get imputation values from training dataset
cate_idx = np.where(np.isin(dataset.columns.values, np.array(one_hot_encoder_list)))[0]
cts_idx = np.setdiff1d(np.arange(dataset.shape[1]), cate_idx)
continuous_median= dataset.iloc[train_idx,cts_idx].median(axis=0).values
categorical_mode = dataset.iloc[train_idx,cate_idx].mode(axis=0).values
impute_dict = dict(zip(dataset.columns.values[cate_idx],categorical_mode.reshape(cate_idx.shape)))
impute_dict.update(dict(zip(dataset.columns.values[cts_idx],continuous_median.reshape(cts_idx.shape))))
# fill back the imputed values
dataset.fillna(impute_dict, inplace=True)

dataset = one_hot_encoder(dataset, encode=one_hot_encoder_list)

encoded_indices = one_hot_indices(dataset, one_hot_encoder_list)
# print("data description:{}".format(dataset.describe()))
covariates = np.array(dataset.columns.values)
# print("columns:{}".format(covariates))
x = np.array(dataset).reshape(dataset.shape)
t = np.array(t_data).reshape(len(t_data))
e = np.array(e_data).reshape(len(e_data))
pat = np.array(pat_data).reshape(len(pat_data))
# print("x:{}, t:{}, e:{}, len:{}".format(x[0], t[0], e[0], len(t)))

print("x_shape:{}".format(x.shape))
# here idx has been shuffled
x = x[idx]
t = t[idx]
e = e[idx]
pat = pat[idx]
end_time = max(t)
print("end_time:{}".format(end_time))
print("observed percent:{}".format(sum(e) / len(e)))
# print("shuffled x:{}, t:{}, e:{}, len:{}".format(x[0], t[0], e[0], len(t)))

print("test:{}, valid:{}, train:{}, all: {}".format(len(test_idx), len(valid_idx), num_examples,
                                                    len(test_idx) + len(valid_idx) + num_examples))
# print("test_idx:{}, valid_idx:{},train_idx:{} ".format(test_idx, valid_idx, train_idx))
train = formatted_data(x=x, t=t, e=e, pat = pat ,idx=train_idx)
test = formatted_data(x=x, t=t, e=e, pat = pat ,idx=test_idx)
valid = formatted_data(x=x, t=t, e=e, pat = pat ,idx=valid_idx)

cat_covariates = np.array(flatten_nested(encoded_indices))
cts_covariates = np.setdiff1d(np.arange(len(covariates)), cat_covariates)
# normalize inputs
norm_mean = np.nanmean(train['x'][:,cts_covariates],axis=0)
norm_std = np.nanstd(train['x'][:,cts_covariates],axis=0)

def saveDatadic(file_path, name, dataset):
    np.save(file_path+name+'_x', dataset['x'])
    np.save(file_path+name+'_t', dataset['t'])
    np.save(file_path+name+'_e', dataset['e'])
    
saveDatadic(file_path, 'flchain_train', train)
saveDatadic(file_path, 'flchain_valid', valid)
saveDatadic(file_path, 'flchain_test', test)

np.save(file_path+'flchain_encoded_indices', encoded_indices)
np.save(file_path+'flchain_covariates', covariates)


## Model hyperparameters
m=100
num_sample = 100
ncov = train['x'].shape[1]
w_e = 1.0
w_ne = 1.0




# split training time based on bins
nbin=100
tt = np.percentile(train['t'][train['e']==1],np.linspace(0.,100.,nbin, endpoint=True))
# based on whether we have censoring after the largest observed t
loss_of_info = np.mean(train['t']>np.max(train['t'][train['e']==1]))

# need to convert t to different size of bins
if loss_of_info > 0.0001:
    nbin = nbin + 1
    # add the largest observed censoring time inside
    tt = np.append(tt,np.max(train['t']))
    event_tt_prob = risk_t_bin_prob(train['t'], train['e'], tt)
    
else:
    # get empirical event rate for re-weighting censoring objects
    event_tt_bin, event_tt_prob = risk_t_bin_prob(train['t'], train['e'], tt)



# define encoder and decoder
slim = tf.contrib.slim
sample_size = 50
# start with 3 layers each
def encoder0(x,is_training):
    """learned prior: Network p(z|x)"""
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.leaky_relu,
#                         normalizer_fn=slim.batch_norm,
#                         normalizer_params={'is_training': is_training},
                    weights_initializer=tf.contrib.layers.xavier_initializer()):

        mu_logvar = slim.fully_connected(x, 64, scope='fc1')
        mu_logvar = slim.fully_connected(mu_logvar, 64, scope='fc2')
        mu_logvar = slim.fully_connected(mu_logvar, 64, activation_fn=None, scope='fc3')
        
    return mu_logvar


def encoder(x,t_, is_training):
    """Network q(z|x,t_)"""
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.leaky_relu,
#                         normalizer_fn=slim.batch_norm,
#                         normalizer_params={'is_training': is_training},
                    weights_initializer=tf.contrib.layers.xavier_initializer()):
        inputs = tf.concat([t_,x],axis=1)
        mu_logvar = slim.fully_connected(inputs, 64, scope='fc1')
        mu_logvar = slim.fully_connected(mu_logvar, 64, scope='fc2')
        mu_logvar = slim.fully_connected(mu_logvar, 64, activation_fn=None, scope='fc3')
        
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
                        activation_fn=tf.nn.leaky_relu,
#                         normalizer_fn=slim.batch_norm,
#                         normalizer_params={'is_training': is_training},
                    weights_initializer=tf.contrib.layers.xavier_initializer()):
        t_logits = slim.fully_connected(z, 64, scope='fc1')
        t_logits = slim.fully_connected(t_logits, 64, scope='fc2')
        t_logits = slim.fully_connected(t_logits, 64, scope='fc3')
       
        # returns multinomial distribution
        t_logits = slim.fully_connected(t_logits, nbin, activation_fn=None, scope='fc4')
        # t_logits = tf.nn.softmax(t_logits)
        
    return (t_logits)

def VAE_losses(t_logits, t_truncate, mu_logvar0, mu_logvar1, tiny=1e-8):
    # NEW ONE! with different strategy of calculating loss for censoring, adding \sum p_b, not \sum w_b*p_b
    """Define loss functions (reconstruction, KL divergence) and optimizer"""
    # Reconstruction loss
    t_dist = tf.nn.softmax(t_logits)
    reconstruction = -tf.log(tf.reduce_sum(t_dist*t_truncate, axis=1))

    # KL divergence
    mu0, logvar0 = tf.split(mu_logvar0, num_or_size_splits=2, axis=1)
    mu1, logvar1 = tf.split(mu_logvar1, num_or_size_splits=2, axis=1)

    kl_d = 0.5 * tf.reduce_sum(tf.exp(logvar1-logvar0)\
                               + tf.divide(tf.square(mu0-mu1),tf.exp(logvar0)+tiny) \
                               + logvar0 - logvar1 -1.0, \
                               1)

    # Total loss for event
    loss = tf.reduce_mean(reconstruction + kl_d)        
    
    return reconstruction, kl_d, loss


def pt_x(t_truncate, mu_logvar0, mu_logvar, num_sample, is_training):
    # here t_ is known!
    # for calculation purposes, censoring subject t_ need to be a truncated form like [0,0,0,1,1,1]
    # which could calculete sum of all bins after censoring time
    mu, logvar = tf.split(mu_logvar0, num_or_size_splits=2, axis=1)
    # sample z_l
    # q_{\beta}(z_l|t_i,x_i)
    epsilon = tf.random_normal(tf.shape(logvar))
    z1_sample = encoder_z(mu_logvar, epsilon)
    # only have one dimension here
    t_logits_l = decoder(z1_sample, is_training)
    t_dist_l = tf.nn.softmax(t_logits_l)    
    p_t_z = tf.reduce_sum(t_truncate*t_dist_l,1)
    pq_z = tf.exp(MVNloglikeli(z1_sample, mu_logvar0, noise = 1e-8)\
                      -MVNloglikeli(z1_sample, mu_logvar, noise = 1e-8))
    pt_x_l = p_t_z*pq_z
    pt_x_sum = pt_x_l
    
    for k in range(num_sample-1):
        # q_{\beta}(z_l|t_i,x_i)
        epsilon = tf.random_normal(tf.shape(logvar))
        z1_sample = encoder_z(mu_logvar, epsilon)
#         # p_{\alpha}(t_i|z_l)
#         epsilon = tf.random_normal(tf.shape(logvar))
#         z0_sample = encoder_z(mu_logvar0, epsilon)
#         # p_{\alpha}(z_l|x)
#         epsilon = tf.random_normal(tf.shape(logvar))
#         # only have one dimension here
        t_logits_l = decoder(z1_sample, is_training)
        t_dist_l = tf.nn.softmax(t_logits_l)    
        p_t_z = tf.reduce_sum(t_truncate*t_dist_l,1)
        pq_z = tf.exp(MVNloglikeli(z1_sample, mu_logvar0, noise = 1e-8)\
                      -MVNloglikeli(z1_sample, mu_logvar, noise = 1e-8))
        pt_x_l = p_t_z*pq_z

        # sum up
        pt_x_sum = pt_x_sum+pt_x_l
        
    pt_x_avg = pt_x_sum/num_sample
    return(pt_x_avg)

def loglikeli_cVAE(t_truncate, mu_logvar0, mu_logvar, num_sample, is_training):
    pt_x_avg = pt_x(t_truncate, mu_logvar0, mu_logvar, num_sample, is_training)
    return(tf.log(pt_x_avg))

# MVN log-likelihood
def MVNloglikeli(z, mu_logvar, noise = 1e-8):
    # Interpret z as concatenation of mean and log variance
    mu, logvar = tf.split(mu_logvar, num_or_size_splits=2, axis=1)

    # note that Sigma is a diagonal matrix and we only have the diagonal information here
    varmatrix = tf.exp(logvar)
    
    # calculate log-likelihood
#     likeli = -0.5*(tf.log(tf.linalg.det(varmatrix)+noise)\
#                    +tf.matmul(tf.matmul((z-mu), tf.linalg.inv(varmatrix))\
#                              ,tf.transpose(z-mu))\
#                    +nbin*np.log(2*np.pi)
#                   )
    # for diagonal matrix:
    loglikeli = -0.5*(tf.log(varmatrix) + (z-mu)**2/varmatrix + np.log(2*np.pi))
    # returns a log-likelihood for each z
    return tf.reduce_sum(loglikeli, axis=1)

def t_dist_avg(mu_logvar0, t_logits_init, num_sample, is_training):
    mu, logvar = tf.split(mu_logvar0, num_or_size_splits=2, axis=1)
    t_dist_new_sum = tf.nn.softmax(t_logits_init)
    for k in range(num_sample-1):
        # graph resample basic implementation
        epsilon = tf.random_normal(tf.shape(logvar))
        t_logits_new_k = decoder(encoder_z(mu_logvar0, epsilon), is_training)
        t_dist_new_k = tf.nn.softmax(t_logits_new_k)
        t_dist_new_sum = t_dist_new_sum +  t_dist_new_k
    t_dist_new_avg = np.divide(t_dist_new_sum, num_sample)   
    return(t_dist_new_avg)

def zero_outputs():
    # just to return 3 outputs to match previous function for events instead
    return 0.0,0.0,0.0

####Main Structure
# training indicator
is_training = tf.placeholder(tf.bool, [], name="is_training");

# Define input placeholder
t_ = tf.placeholder(tf.float32,[None, nbin], name='t_')
# Define input placeholder only for calculating likelihood or survival function purpose
t_truncate = tf.placeholder(tf.float32,[None, nbin], name='t_truncate')

# each patient will only have 1 indicator of censoring or event
event = tf.placeholder(tf.float32,[None], name='event')
x = tf.placeholder(tf.float32,[None, ncov], name='x')

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
    
    # calculate likelihood based on randomly sample multiple z1
    event_loglikeli = loglikeli_cVAE(tf.gather(t_truncate,e_idx), tf.gather(mu_logvar0,e_idx), tf.gather(mu_logvar1,e_idx), num_sample, is_training)
    censor_loglikeli = loglikeli_cVAE(tf.gather(t_truncate,ne_idx), tf.gather(mu_logvar0,ne_idx), tf.gather(mu_logvar1,ne_idx), num_sample, is_training)

    total_loglikeli = loglikeli_cVAE(t_truncate, mu_logvar0, mu_logvar1, num_sample, is_training)
# Optimization
with tf.variable_scope('training') as scope:
    # calculate the losses separately, just for debugging purposes
    # calculate losses for events
    e_recon, e_kl_d, eloss = tf.cond(e_is_empty, lambda: zero_outputs(),\
                                     lambda:VAE_losses(tf.gather(t_logits_1,e_idx), tf.gather(t_truncate,e_idx), \
                         tf.gather(mu_logvar0,e_idx), tf.gather(mu_logvar1,e_idx)))
    
    # calculate losses for censor
    ne_recon, ne_kl_d, closs = tf.cond(ne_is_empty, lambda: zero_outputs(),\
                                       lambda: VAE_losses(tf.gather(t_logits_1,ne_idx), tf.gather(t_truncate,ne_idx), \
                         tf.gather(mu_logvar0,ne_idx), tf.gather(mu_logvar1,ne_idx)))
                                       
    loss = w_e*eloss+w_ne*closs
    # compute together
    rec_all, kl_d_all, loss_all = VAE_losses(t_logits_1,t_truncate, mu_logvar0, mu_logvar1)
#    train_step_unlabeled = tf.train.AdamOptimizer().minimize(loss)
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    gradients = tf.gradients(loss_all, params)
    #gradients = tf.Print(gradients,[gradients], message ='gradients',summarize=2000)
    grads = zip(gradients, params) 
    
    optimizer = tf.train.AdamOptimizer(learning_rate=5e-4, beta1=0.9, beta2=0.999)
    train_step = optimizer.apply_gradients(grads)
    
def wAvg_t(sess, new_x, post_prob, tt, num_sample, return_wi=False):
    # calculate weighted average
    for j in range(num_sample):
        t_hat_l = np.array([random_uniform_p(tt, post_prob[subj], 1) for subj in range(post_prob.shape[0])])
        t_hat_binned = batch_t_categorize(t_hat_l, np.ones(t_hat_l.shape), tt, event_tt_prob=1.0)
        mu_logvar0l = sess.run(mu_logvar0, feed_dict={x: new_x, is_training:False})
        mu_logvar1l = sess.run(mu_logvar1, feed_dict={x: new_x, t_: t_hat_binned,is_training:False})
        # sample z1l
        mu1l,logvar1l = np.split(mu_logvar1l,2,1)
        epsilon_l = np.random.normal(size=logvar1l.shape)
        # Standard deviation must be positive
        stddevl = np.sqrt(np.exp(logvar1l))
        z1l = mu1l + np.multiply(stddevl, epsilon_l)        
        ## calculate weight
        wil = np.divide(np.exp(MVNloglikeli_np(z1l, mu_logvar0l, noise = 1e-8)),\
                                           np.exp(MVNloglikeli_np(z1l, mu_logvar1l, noise = 1e-8)))
        if j == 0:
            t_hat_all = np.array(t_hat_l).reshape(post_prob.shape[0],1)
            wl_all = wil.reshape(post_prob.shape[0],1)
        else:
            t_hat_all = np.concatenate([t_hat_all, np.array(t_hat_l).reshape(post_prob.shape[0],1)], axis=1)
            wl_all = np.concatenate([wl_all, wil.reshape(post_prob.shape[0],1)], axis=1)

    t_hat_i = np.sum(np.multiply(t_hat_all,wl_all),axis=1)/np.sum(wl_all,axis=1)
    if return_wi==False:
        return t_hat_i
    else:
        return (t_hat_i, np.mean(wl_all, axis=1), np.std(wl_all, axis=1))

    
def saveResults(dataset, session_dir, session_name, out_dir, tt, event_tt_prob):
    sess = tf.Session()
    session_path = session_dir+session_name+".ckpt"
    saver.restore(sess,  session_path)
    # run over all samples in test
    batch_x, batch_t, batch_e = dataset['x'], dataset['t'], dataset['e']
    batch_t_cat = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob)

    batch_t_cat_likeli = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob,likelihood=True)
    norm_batch_x = batch_x.copy()
    norm_batch_x[:,cts_covariates] = normalize_batch(batch_x[:,cts_covariates],norm_mean,norm_std)
    test_pred_prob = sess.run(t_dist_new_avg, feed_dict={x: norm_batch_x, is_training:False})
    test_loglikeli = sess.run(total_loglikeli, feed_dict={t_truncate:batch_t_cat_likeli, t_:batch_t_cat, x:norm_batch_x, event:batch_e, is_training:False})
    # this provide likelihood
#     test_pt_x_avg = sess.run(total_pt_x_avg, feed_dict={t_truncate:batch_t_cat_likeli, t_:batch_t_cat, x:batch_x, event:batch_e, is_training:False})
    test_pred_avgt, test_avgt_mean, test_avgt_std = wAvg_t(sess, norm_batch_x, test_pred_prob, tt, num_sample, return_wi=True)

    test_pred_medt = [calculate_quantiles(post_prob,tt,0.5) for post_prob in test_pred_prob]
    test_pred_medt = np.concatenate(test_pred_medt,axis=0)
    test_pred_randomt = np.array([random_uniform_p(tt, post_prob, 1) for post_prob in test_pred_prob])
    np.save(out_dir+'/{}_test_pred_prob'.format(session_name), test_pred_prob)
    np.save(out_dir+'/{}_test_loglikeli'.format(session_name), test_loglikeli)
    np.save(out_dir+'/{}_test_pred_avgt'.format(session_name), test_pred_avgt)
    np.save(out_dir+'/{}_test_pred_medt'.format(session_name), test_pred_medt)
    np.save(out_dir+'/{}_test_pred_randomt'.format(session_name), test_pred_randomt)
    np.save(out_dir+'/{}_tt'.format(session_name), tt)

def saveResults_norun(session_name, out_dir, tt, test_pred_prob, test_loglikeli, test_pred_avgt, test_pred_medt, test_pred_randomt):
    np.save(out_dir+'/{}_test_pred_prob'.format(session_name), test_pred_prob)
    np.save(out_dir+'/{}_test_loglikeli'.format(session_name), test_loglikeli)
    np.save(out_dir+'/{}_test_pred_avgt'.format(session_name), test_pred_avgt)
    np.save(out_dir+'/{}_test_pred_medt'.format(session_name), test_pred_medt)
    np.save(out_dir+'/{}_test_pred_randomt'.format(session_name), test_pred_randomt)
    np.save(out_dir+'/{}_tt'.format(session_name), tt)

    
##########################
#### Training ############
##########################
if training==True:
    valid_recon_loss = []
    valid_epoch_recon_loss = []
    valid_epoch_loss = []
    valid_epoch_event_recon_loss = []
    valid_epoch_censor_recon_loss = []

    best_likelihood = -np.inf
    best_i = 0
    best_epoch = 0
    num_epoch = 200
    num_sample = 100
    num_batch = int(train['x'].shape[0]/m)
    require_impr = 3000
    saver = tf.train.Saver()
    # event_tt_prob = event_t_bin_prob_unif(tt)
    with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Train VAE model
        for i in range(num_epoch*num_batch):        
            # Get a training minibatch
            batch_x, batch_t, batch_e = next_batch(train, m=m)
            batch_t_cat_likeli = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob,likelihood=True)
            # normalize input
            norm_batch_x = batch_x.copy()
            norm_batch_x[:,cts_covariates] = normalize_batch(batch_x[:,cts_covariates],norm_mean,norm_std)
            # Binarize the data
            batch_t_cat = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob)
            # Train on minibatch
            sess.run(train_step, feed_dict={t_:batch_t_cat, t_truncate: batch_t_cat_likeli, x:norm_batch_x, event:batch_e, is_training:True})
            # sess.run(train_step_SGD, feed_dict={t_:batch_t_cat, x:batch_x, event:batch_e, is_training:True})

            if i % num_batch == 0:
                batch_x, batch_t, batch_e = next_batch(valid, m=valid['x'].shape[0])
                batch_t_cat = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob)
                batch_t_cat_likeli = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob,likelihood=True)
                norm_batch_x = batch_x.copy()
                norm_batch_x[:,cts_covariates] = normalize_batch(batch_x[:,cts_covariates],norm_mean,norm_std)

                epoch_loglikeli = np.mean(sess.run(total_loglikeli, feed_dict={t_:batch_t_cat, t_truncate: batch_t_cat_likeli,\
                                                                            x: norm_batch_x, event:batch_e, is_training:False}))
                epoch_loss = sess.run(loss_all, feed_dict={t_:batch_t_cat, t_truncate: batch_t_cat_likeli, x: norm_batch_x, event:batch_e, is_training:False})

                valid_epoch_recon_loss.append(epoch_loglikeli)
                valid_epoch_loss.append(epoch_loss)
                epoch_recon_closs = np.mean(sess.run(ne_recon, feed_dict={t_:batch_t_cat, t_truncate: batch_t_cat_likeli, x: norm_batch_x, event:batch_e, is_training:False}))
                valid_epoch_censor_recon_loss.append(epoch_recon_closs)
                epoch_recon_eloss = np.mean(sess.run(e_recon, feed_dict={t_:batch_t_cat, t_truncate: batch_t_cat_likeli, x: norm_batch_x, event:batch_e, is_training:False}))
                valid_epoch_event_recon_loss.append(epoch_recon_eloss)
                if (best_likelihood <= epoch_loglikeli):
                    best_likelihood = epoch_loglikeli
                    best_i = i
                    # save the learned model
                    save_path = saver.save(sess, output_dir+name+".ckpt")

                op_print = ('Epoch '+str(i/num_batch)+': Loss '+str(epoch_loss)\
                      +' log-likelihood: ' + str(epoch_loglikeli)\
                      +' event rec loss: ' + str(epoch_recon_eloss)\
                      +' censor rec loss: ' + str(epoch_recon_closs))
                logging.debug(op_print)

            # early stopping    
            if (i-best_i) > require_impr:
                print("Model stops improving for a while")
                break
    ##### return results on testing dataset #####
    # run over all samples in test
    saveResults(test, session_dir=output_dir, session_name=name, out_dir=out_dir, tt=tt, event_tt_prob=event_tt_prob)
    

                
#### only for testing #####
else:
    sess = tf.Session()
    # Restore variables from disk.
    saver.restore(sess,  output_dir+name+".ckpt")
    # run over all samples in test

    # run over all samples in test
    batch_x, batch_t, batch_e = test['x'], test['t'], test['e']
    batch_t_cat = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob)

    batch_t_cat_likeli = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob,likelihood=True)

    norm_batch_x = batch_x.copy()
    norm_batch_x[:,cts_covariates] = normalize_batch(batch_x[:,cts_covariates],norm_mean,norm_std)
    test_pred_prob = sess.run(t_dist_new_avg, feed_dict={x: norm_batch_x, is_training:False})
    test_loglikeli = sess.run(total_loglikeli, feed_dict={t_truncate:batch_t_cat_likeli, t_:batch_t_cat, x:norm_batch_x, event:batch_e, is_training:False})
    test_pred_avgt, test_avgt_mean, test_avgt_std = wAvg_t(sess, norm_batch_x, test_pred_prob, tt, num_sample, return_wi=True)

    test_pred_medt = [calculate_quantiles(post_prob,tt,0.5) for post_prob in test_pred_prob]
    test_pred_medt = np.concatenate(test_pred_medt,axis=0)
    test_pred_randomt = np.array([random_uniform_p(tt, post_prob, 1) for post_prob in test_pred_prob])
    
    saveResults_norun(session_name=name, out_dir=out_dir, tt=tt, test_pred_prob=test_pred_prob, test_loglikeli=test_loglikeli, test_pred_avgt=test_pred_avgt, test_pred_medt=test_pred_medt, test_pred_randomt=test_pred_randomt)
    