import numpy as np
import tensorflow as tf
from utils.preprocessing import batch_t_categorize

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

def calculate_quantiles(post_prob, tt, percentiles):
    post_prob_sum = np.cumsum(post_prob)
    try:
        tt_p = [tt[np.argmin(np.abs(post_prob_sum-p))] for p in percentiles]
    except TypeError:
        tt_p = tt[np.argmin(np.abs(post_prob_sum-percentiles))]
        tt_p = [tt_p]
        
    return(np.array(tt_p))

def random_multinomial(tt, pred_prob, n_sample):
    select_idx = np.argmax(np.random.multinomial(n_sample, pred_prob, size=1))
    return tt[select_idx]

def random_uniform_p(tt, pred_prob, n_sample):
    select_idx = np.max(np.random.choice(len(tt), n_sample, p=pred_prob))
    return tt[select_idx]

def event_likelihood(new_t, new_e, pred_prob, tt, event_tt_prob, tiny=0):
    if np.array(new_t).size==1 and new_e==1:
        obs_likeli = np.sum(batch_t_categorize(new_t.reshape([1,1]), new_e.reshape([1,1]), tt, event_tt_prob)\
                            *np.log(pred_prob+tiny))
        
    else:
        event_idx = np.where(new_e==1)
        new_t_cat = batch_t_categorize(new_t[event_idx], new_e[event_idx], tt, event_tt_prob)
        obs_likeli = np.sum(np.multiply(new_t_cat, np.log(pred_prob[event_idx]+tiny)), axis=1)
    return obs_likeli

def censor_likelihood(new_t, new_e, pred_prob, tt, event_tt_prob, tiny=0):
    if np.array(new_t).size==1 and new_e==0:
        new_t_cat = batch_t_categorize(new_t.reshape([1,1]), new_e.reshape([1,1]), tt, event_tt_prob)
        new_t_cat = 1*(new_t_cat>0)
        cen_likeli = np.log(np.sum(np.multiply(new_t_cat, pred_prob))+tiny)
        
    else:
        cen_idx = np.where(new_e==0)
        new_t_cat = batch_t_categorize(new_t[cen_idx], new_e[cen_idx], tt, event_tt_prob)
        new_t_cat = 1*(new_t_cat>0)
        cen_likeli = np.log(np.sum(np.multiply(new_t_cat, pred_prob[cen_idx]), axis=1)+tiny)
    return cen_likeli

def MVNloglikeli_np(z, mu_logvar, noise = 1e-8):
    # Interpret z as concatenation of mean and log variance
    mu, logvar = np.split(mu_logvar, 2, 1)

    # note that Sigma is a diagonal matrix and we only have the diagonal information here
    varmatrix = np.exp(logvar)
    
    # calculate log-likelihood
#     likeli = -0.5*(tf.log(tf.linalg.det(varmatrix)+noise)\
#                    +tf.matmul(tf.matmul((z-mu), tf.linalg.inv(varmatrix))\
#                              ,tf.transpose(z-mu))\
#                    +nbin*np.log(2*np.pi)
#                   )
    # for diagonal matrix:
    loglikeli = -0.5*(np.log(varmatrix+noise) + (z-mu)**2/varmatrix + np.log(2*np.pi))
    # returns a log-likelihood for each z
    return np.sum(loglikeli, axis=1)


def err_dist_non_censor(event, t_, pred_t, truncate_t=np.inf):
    event = (event==1) & (t_<truncate_t)
    err_dist = np.divide(pred_t[event] - t_[event],np.max(t_))
    err_mean = np.mean(np.divide(np.abs(pred_t[event] - t_[event]),np.max(t_)))
    return(err_mean, err_dist)


def err_dist_censor(event, t_, pred_t, truncate_t):
    event = (event==0) |( (event==1) & (t_>truncate_t))
    
#     return(np.divide(np.minimum((pred_t[event] - t_[event]),\
#                                      np.zeros(len(t_[event]))),np.max(t_)))
    err_dist = np.divide((pred_t[event] - t_[event]),np.max(t_))
    err_mean = np.mean(np.minimum(np.divide(np.abs(pred_t[event] - t_[event]),np.max(t_)),\
                              np.zeros(pred_t[event].shape)))
    return(err_mean, err_dist)

def I_Ctd_cVAE(dataset, test_pred_prob, tt, i,j):
    x_i = dataset['x'][i][-1]
    x_j = dataset['x'][j][-1]
    t_true_i = dataset['t'][i]
    t_i_idx = np.where(batch_t_categorize(dataset['t'][i].reshape([1,1]), dataset['e'][i].reshape([1,1]), tt, event_tt_prob,likelihood=True)[-1]==1)[0]
    sum_idx = np.concatenate([np.ones(t_i_idx), np.zeros(len(tt)-t_i_idx)])
    F_i = np.dot(test_pred_prob[i], sum_idx)
    F_j = np.dot(test_pred_prob[j], sum_idx)
    return(1*(F_i > F_j))
    # return (log_S_i, log_S_j)

def pair_Ctd_cVAE(dataset):
    subj_i = np.random.choice(np.where(dataset['e']==1)[0],1)
    subj_j = np.random.choice(np.where(dataset['t']>dataset['t'][subj_i])[0],1)
    return(I_Ctd_cVAE(dataset, test_pred_prob, tt, subj_i,subj_j))

def I_Ctd_AFT(dataset, i,j):
    x_i = dataset['x'][i][-1]
    x_j = dataset['x'][j][-1]
    t_true_i = dataset['t'][i]
    beta_xi = b0 + b1*x_i[0] + b2*x_i[1]
    beta_xj = b0 + b1*x_j[0] + b2*x_j[1]
    
    log_S_i = -(t_true_i/np.exp(beta_xi))**rho
    log_S_j = -(t_true_i/np.exp(beta_xj))**rho
    return(1*(log_S_i <= log_S_j))
    # return (log_S_i, log_S_j)

def pair_Ctd_AFT(dataset):
    subj_i = np.random.choice(np.where(dataset['e']==1)[0],1)
    subj_j = np.random.choice(np.where(dataset['t']>dataset['t'][subj_i])[0],1)
    return(I_Ctd_AFT(dataset, subj_i,subj_j))

def KS_D(F1, F2, tt):
    diff_list = np.abs((np.ones(len(tt))-F1)-(np.ones(len(tt))-F2))
    return(tt[np.argmax(diff_list)], max(diff_list), diff_list)
