import tensorflow as tf
import numpy as np

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

def t_tt_idx(t_,tt):
    t_diff = t_-tt
    t_diff_sign = (t_diff>=0)
    if len(t_diff[t_diff_sign])>0:
        idx = np.argmin(t_diff[t_diff_sign])
    else:
        idx = 0
    return idx

def event_t_bin_prob(batch_t, batch_e, tt):
    nbin = len(tt)
    bin_all = np.arange(nbin)
    # only consider the event time, not including censoring time
    nbatch = batch_t[batch_e==1].shape[0]
    tt_idx = [t_tt_idx(t_obs,tt) for t_obs in batch_t[batch_e==1]]
    bin_unique, bin_count = np.unique(tt_idx, return_counts=True)
    
    # fill back the unique values not included with 0 probabilities
    bin_left = np.setdiff1d(bin_all, bin_unique)
    bin_all = np.concatenate([bin_unique,bin_left])
    bin_count = np.concatenate([bin_count,np.zeros(len(bin_left))])
    
    # sort based on idx
    sort_idx = np.argsort(bin_all)
    return bin_all[sort_idx],bin_count[sort_idx]/nbatch

def event_t_bin_prob_unif(tt):
    nbin = len(tt)
    return np.repeat(1/nbin, nbin)


def next_batch(dataset, m=100):
    idx = np.random.choice(dataset['x'].shape[0], m)
    return((dataset['x'][idx,:], dataset['t'][idx], dataset['e'][idx]))

# transform t into categories
# one-hot-encoding
# only consider the nearest category to the left
def t_label(t_,tt):
    nbin = len(tt)
    t_diff = t_-tt
    t_diff_sign = (t_diff>=0)
    if len(t_diff[t_diff_sign])>0:
        idx = np.argmin(t_diff[t_diff_sign])
        t_cat = np.zeros(nbin)
        t_cat[idx] = 1.0
    else:
        t_cat = np.zeros(nbin)
        t_cat[0] = 1.0
    return t_cat

def t_reweight(t_, tt, event_tt_prob):
    t_idx = t_tt_idx(t_,tt)
    t_raw_prob = event_tt_prob[np.arange(t_idx, len(tt))]
    t_prob = np.concatenate([np.zeros(t_idx), t_raw_prob/np.sum(t_raw_prob)])
    return t_prob
    
def t_categorize(t_, e_, tt, event_tt_prob):
    if e_ == 1:
        t_cat = t_label(t_,tt)
    else:
        t_cat = t_reweight(t_, tt, event_tt_prob)
    return t_cat
    
def batch_t_categorize(batch_t, batch_e, tt, event_tt_prob):
    nbin = len(tt)
    nbatch = batch_t.shape[0]
    all_cat = np.array([t_categorize(batch_t[obs], batch_e[obs], tt, event_tt_prob) for obs in np.arange(nbatch)])    
    return all_cat

def pred_prob(new_x,num_sample):
#     sess = tf.Session()
#     # Restore variables from disk.
#     saver.restore(sess,  output_dir+name+".ckpt")
    pred_prob_list = []
    # need to run the function within a tf session
    for k in np.arange(num_sample):
        prob_subj = sess.run(t_dist_new, feed_dict={x:new_x.reshape([1,1])}).reshape(-1)
        pred_prob_list.append(prob_subj)
        
    pred_prob = np.mean(np.array(pred_prob_list), axis=0)
    return pred_prob
