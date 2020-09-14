import math
import os
import numpy as np
import pandas
import tensorflow as tf

# one-hot-encoding all categorical variables
def one_hot_encoder(data, encode):
    data_encoded = data.copy()
    encoded = pandas.get_dummies(data_encoded, prefix=encode, columns=encode)
#    print("head of data:{}, data shape:{}".format(data_encoded.head(), data_encoded.shape))
#    print("Encoded:{}, one_hot:{}{}".format(encode, encoded.shape, encoded[0:5]))
    return encoded
# return column indices for one columns
def one_hot_indices(dataset, one_hot_encoder_list):
    indices_by_category = []
    for column in one_hot_encoder_list:
        values = dataset.filter(regex="{}_.*".format(column)).columns.values
        # print("values:{}".format(values, len(values)))
        indices_one_hot = []
        for value in values:
            indice = dataset.columns.get_loc(value)
            # print("column:{}, indice:{}".format(colunm, indice))
            indices_one_hot.append(indice)
        indices_by_category.append(indices_one_hot)
    # print("one_hot_indices:{}".format(indices_by_category))
    return indices_by_category

def get_train_median_mode(x, categorial):
    categorical_flat = flatten_nested(categorial)
    print("categorical_flat:{}".format(categorical_flat))
    imputation_values = []
    print("len covariates:{}, categorical:{}".format(x.shape[1], len(categorical_flat)))
    median = np.nanmedian(x, axis=0)
    mode = []
    for idx in np.arange(x.shape[1]):
        a = x[:, idx]
        (_, idx, counts) = np.unique(a, return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]
        mode_idx = a[index]
        mode.append(mode_idx)
    for i in np.arange(x.shape[1]):
        if i in categorical_flat:
            imputation_values.append(mode[i])
        else:
            imputation_values.append(median[i])
    print("imputation_values:{}".format(imputation_values))
    return imputation_values


def missing_proportion(dataset):
    missing = 0
    columns = np.array(dataset.columns.values)
    for column in columns:
        missing += dataset[column].isnull().sum()
    return 100 * (missing / (dataset.shape[0] * dataset.shape[1]))

# for training/testing/validation split
def formatted_data_original(x, t, e, idx):
    death_time = np.array(t[idx], dtype=float)
    censoring = np.array(e[idx], dtype=float)
    covariates = np.array(x[idx])

    print("observed fold:{}".format(sum(e[idx]) / len(e[idx])))
    survival_data = {'x': covariates, 't': death_time, 'e': censoring}
    return survival_data

def shape_train_valid(x,train_idx, valid_idx):
    train = np.array(x[train_idx])
    valid = np.array(x[valid_idx])

    train_cov = np.where((train != 0).any(axis=0))[0]
    valid_cov = np.where((valid != 0).any(axis=0))[0]
    train_valid = np.intersect1d(train_cov, valid_cov)
    
    return(train_valid)

def formatted_data(x, t, e, pat,idx):
    death_time = np.array(t[idx], dtype=float)
    censoring = np.array(e[idx], dtype=float)
    covariates = np.array(x[idx])

    #print("observed fold:{}".format(sum(e[idx]) / len(e[idx])))
    survival_data = {'x': covariates, 't': death_time, 'e': censoring, 'pat': np.array(pat[idx], dtype=int)}
    return survival_data

def formatted_data_simu(x, t, e, T, idx):
    death_time = np.array(t[idx], dtype=float)
    censoring = np.array(e[idx], dtype=float)
    covariates = np.array(x[idx])

    #print("observed fold:{}".format(sum(e[idx]) / len(e[idx])))
    survival_data = {'x': covariates, 't': death_time, 'e': censoring, 'T':np.array(T[idx])}
    return survival_data 


# 
def flatten_nested(list_of_lists):
    flattened = [val for sublist in list_of_lists for val in sublist]
    return flattened

# z-normalization to input
def normalize_batch(batch_x,norm_mean,norm_std):
    return((batch_x-norm_mean)/norm_std)

def get_missing_mask(data, imputation_values=None):
    copy = data
    for i in np.arange(len(data)):
        row = data[i]
        indices = np.isnan(row)
        # print("indices:{}, {}".format(indices, np.where(indices)))
        if imputation_values is None:
            copy[i][indices] = 0
        else:
            for idx in np.arange(len(indices)):
                if indices[idx]:
                    # print("idx:{}, imputation_values:{}".format(idx, np.array(imputation_values)[idx]))
                    copy[i][idx] = imputation_values[idx]
    # print("copy;{}".format(copy))
    return copy

def risk_set(data_t):
    size = len(data_t)
    risk_set = np.zeros(shape=(size, size))
    for idx in range(size):
        temp = np.zeros(shape=size)
        t_i = data_t[idx]
        at_risk = data_t > t_i
        temp[at_risk] = 1
        # temp[idx] = 0
        risk_set[idx] = temp
    return risk_set

##### for re-formulate time #####
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

def risk_t_bin_prob(batch_t, batch_e, tt):
    # using counting table idea
    # Start from NA estimater of Ht, then get empirical ft accordingly
    nbin = len(tt)
    bin_all = np.arange(nbin)

    risk_count_all = []
    # calculate at each t in tt, the corresponding dj
    tt_idx = [t_tt_idx(t_obs,tt) for t_obs in batch_t[batch_e==1]]
    bin_unique, bin_count = np.unique(tt_idx, return_counts=True)
    # fill back the unique values not included with 0 probabilities
    bin_left = np.setdiff1d(bin_all, bin_unique)
    bin_all = np.concatenate([bin_unique,bin_left])
    bin_count = np.concatenate([bin_count,np.zeros(len(bin_left))])
    
    # calculate nj
    for i in bin_all:
        t_i = tt[i]
        risk_count = (np.where(batch_t>=t_i)[0]).shape[0]
        risk_count_all.append(risk_count)
    
    # sort based on idx
    sort_idx = np.argsort(bin_all)
    
    d_n = bin_count[sort_idx]/risk_count_all
    Ht = np.cumsum(d_n)
    
    St = np.exp(-Ht)
    ft = -np.diff(St)
    f0 = np.array([1-St[0]])
    ft[-1] = 1-np.sum(ft)-f0
    event_tt_prob = np.concatenate([f0,ft])
    
    return event_tt_prob



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

def t_reweight(t_, tt, event_tt_prob, likelihood=False):
    t_idx = t_tt_idx(t_,tt)
    if not likelihood:
        t_raw_prob = event_tt_prob[np.arange(t_idx, len(tt))]
        t_prob = np.concatenate([np.zeros(t_idx), t_raw_prob/np.sum(t_raw_prob)])
    else:
        t_prob = np.concatenate([np.zeros(t_idx), np.ones(len(tt)-t_idx)])
    return t_prob
    
def t_categorize(t_, e_, tt, event_tt_prob,likelihood=False):
    if e_ == 1:
        t_cat = t_label(t_,tt)
    else:
        t_cat = t_reweight(t_, tt, event_tt_prob,likelihood)
    return t_cat
    
def batch_t_categorize(batch_t, batch_e, tt, event_tt_prob,likelihood=False):
    nbin = len(tt)
    nbatch = batch_t.shape[0]
    all_cat = np.array([t_categorize(batch_t[obs], batch_e[obs], tt, event_tt_prob,likelihood) for obs in np.arange(nbatch)])    
    return all_cat


def t_label_binned(t_,tt):
    nbin = len(tt)
    t_diff = t_-tt
    t_diff_sign = (t_diff>=0)
    if len(t_diff[t_diff_sign])>0:
        idx = np.argmin(t_diff[t_diff_sign])
    else:
        idx = 0
    return tt[idx]

def discretized_t_categorize(batch_t, tt):
    nbin = len(tt)
    nbatch = batch_t.shape[0]
    all_cat = np.array([t_label_binned(batch_t[obs], tt) for obs in np.arange(nbatch)])    
    return all_cat


def log_transform(data, transform_ls):
    dataframe_update = data

    def transform(x):
        constant = 1e-8
        transformed_data = np.log(x + constant)
        # print("max:{}, min:{}".format(np.max(transformed_data), np.min(transformed_data)))
        return np.abs(transformed_data)

    for column in transform_ls:
        df_column = dataframe_update[column]
        print(" before log transform: column:{}{}".format(column, df_column.head()))
        print("stats:max: {}, min:{}".format(df_column.max(), df_column.min()))
        dataframe_update[column] = dataframe_update[column].apply(transform)
        print(" after log transform: column:{}{}".format(column, dataframe_update[column].head()))
    return dataframe_update

############
def saveDataDict(data_dic, name, file_path):
    x = data_dic['x']
    e = data_dic['e']
    t = data_dic['t']
    pat = data_dic['pat']
    np.save(file_path+'/{}_x'.format(name), x)
    np.save(file_path+'/{}_e'.format(name), e)
    np.save(file_path+'/{}_t'.format(name), t)
    np.save(file_path+'/{}_pat'.format(name), pat)

    
def loadDataDict(name, file_path):
    x = np.load(file_path+'{}_x'.format(name)+'.npy')
    e = np.load(file_path+'{}_e'.format(name)+'.npy')
    t = np.load(file_path+'{}_t'.format(name)+'.npy')
    pat = np.load(file_path+'{}_pat'.format(name)+'.npy')
    return({'x':x, 'e':e, 't':t, 'pat':pat})

def saveDataPreprocessed(data_set, file_path):
    end_t, covariates, one_hot_indices, imputation_values =  data_set['end_t'],\
    data_set['covariates'], \
    data_set['one_hot_indices'], \
    data_set['imputation_values']
    np.save(file_path+'/end_t', end_t)
    np.save(file_path+'/covariates', covariates)
    np.save(file_path+'/one_hot_indices', one_hot_indices)
    np.save(file_path+'/imputation_values', imputation_values)

def loadDataPreprocessed(file_path):
    end_t, covariates, one_hot_indices, imputation_values =  np.load(file_path+'/end_t'+'.npy'),\
    np.load(file_path+'/covariates'+'.npy'),\
    np.load(file_path+'/one_hot_indices'+'.npy'),\
    np.load(file_path+'/imputation_values'+'.npy')
    return (end_t, covariates, one_hot_indices, imputation_values)

def datadicTimeCut(data_dic, time_cut=168, time_cut_low=0):
    keep_idx = np.where((data_dic['t'] > time_cut_low)&(data_dic['t'] <= time_cut))
    return({'x':data_dic['x'][keep_idx], 'e':data_dic['e'][keep_idx], 't':data_dic['t'][keep_idx]})



