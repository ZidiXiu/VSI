import os

import numpy as np
import pandas

from utils.preprocessing import one_hot_encoder, formatted_data, missing_proportion, \
    one_hot_indices, get_train_median_mode, shape_train_valid

def generate_data(file_path):
    # this program is to extract the preset training/testing/validating dataset
    np.random.seed(31415)
    dir_path = os.path.dirname(file_path)
    
    path = os.path.abspath(os.path.join(dir_path, '', 'data.csv'))
    print("path:{}".format(path))
    # row number as index column
    data_frame = pandas.read_csv(path)
    print("head of data:{}, data shape:{}".format(data_frame.head(), data_frame.shape))
    
    path_vital_var = os.path.abspath(os.path.join(dir_path, '', 'vital_var.csv'))
    print("path:{}".format(path_vital_var))
    vital_var = pandas.read_csv(path_vital_var).values.tolist()
    vital_var = [item for sublist in vital_var for item in sublist]
    path_cate_var = os.path.abspath(os.path.join(dir_path, '', 'categorical_var.csv'))
    print("path:{}".format(path_cate_var))
    cate_var = pandas.read_csv(path_cate_var).values.tolist()
    cate_var = [item for sublist in cate_var for item in sublist]
    cts_var = np.setdiff1d(vital_var, cate_var)
    
    # select one observation per-patient
    dataset = data_frame.groupby('PAT_ENC_CSN_ID').first()

    #print("head of data:{}, data shape:{}".format(data_frame.head(), data_frame.shape))
    # x_xdata = data_frame[['age', 'sex', 'kappa', 'lambda', 'flc.grp', 'creatinine', 'mgus']]
    # Preprocess
    to_drop = ['TRAIN_VALIDATION_TEST','OVERALL_OUTCOME','TWO_HOUR_BLOCK_START_TIME','TWO_HOUR_BLOCK_STOP_TIME', 'TWO_HOUR_BLOCK_EVENT_HOURS']
    
    #delete 'TWO_HOUR_BLOCK_EVENT_HOURS'==0
    dataset = dataset[dataset['TWO_HOUR_BLOCK_EVENT_HOURS'] > 0]
    
    print("head of cleaned data:{}, cleaned data shape:{}".format(data_frame.head(), data_frame.shape))
    #data_frame = data_frame.reindex(range(data_frame.shape[0]))
    idx = np.arange(0, dataset.shape[0])
    np.random.shuffle(idx)
    num_examples = int(0.80 * dataset.shape[0])    
    train_idx = idx[0: num_examples]
    split = int((dataset.shape[0] - num_examples) / 2)

    test_idx = idx[num_examples: num_examples + split]
    valid_idx = idx[num_examples + split: dataset.shape[0]]
    
    # impute nan values with train data
    cate_idx = np.where(np.isin(dataset.columns.values, np.array(cate_var)))[0]
    cts_idx = np.where(np.isin(dataset.columns.values, np.array(cts_var)))[0]
    continuous_median= dataset.iloc[train_idx,cts_idx].median(axis=0).values
    categorical_mode = dataset.iloc[train_idx,cate_idx].mode(axis=0).values
    
    impute_dict = dict(zip(dataset.columns.values[cate_idx],categorical_mode.reshape(cate_idx.shape)))
    impute_dict.update(dict(zip(dataset.columns.values[cts_idx],continuous_median.reshape(cts_idx.shape))))
    # fill back the imputed values
    dataset.fillna(impute_dict, inplace=True)
    
    
    print("missing:{}".format(missing_proportion(dataset.drop(labels=to_drop, axis=1))))
    one_hot_encoder_list = cate_var.copy()
    dataset = one_hot_encoder(dataset, encode=one_hot_encoder_list)
    t_data = dataset[['TWO_HOUR_BLOCK_EVENT_HOURS']]
    e_data = dataset[['OVERALL_OUTCOME']]
    pat_data = dataset.index.values
    dataset = dataset.drop(labels=to_drop, axis=1)
    print("head of dataset data:{}, data shape:{}".format(dataset.head(), dataset.shape))

    #print("data description:{}".format(dataset.describe()))
    covariates = np.array(dataset.columns.values)
    print("columns:{}".format(covariates))
    x = np.array(dataset).reshape(dataset.shape)
    t = np.array(t_data).reshape(len(t_data))
    e = np.array(e_data).reshape(len(e_data))
    pat = np.array(pat_data).reshape(len(pat_data))
    
    # check data to make sure no all nan or 0 in all three datasets, delete all 0/NaN columns in training
    covariate_no0_idx = shape_train_valid(x,train_idx, valid_idx)
    covariates_new = covariates[covariate_no0_idx]

    #encoded_indices_new = [list(filter(lambda x: x in covariate_no0_idx, sublist)) for sublist in encoded_indices]

    # subset dataset with the new covariates list
    dataset = dataset.iloc[:,covariate_no0_idx]
    x = np.array(dataset).reshape(dataset.shape)
    
    # update encoded_indices
    encoded_indices = one_hot_indices(dataset, one_hot_encoder_list)

    print("x:{}, t:{}, e:{}, len:{}".format(x[0], t[0], e[0], len(t)))
    print("x_shape:{}".format(x.shape))
    
    train = formatted_data(x=x, t=t, e=e, pat=pat, idx=train_idx)
    test = formatted_data(x=x, t=t, e=e, pat=pat, idx=test_idx)
    valid = formatted_data(x=x, t=t, e=e, pat=pat, idx=valid_idx)
    
    end_time = max(t[e==1])
    print("end_time:{}".format(end_time))
    print("observed percent:{}".format(sum(e) / len(e)))
 
    imputation_values = get_train_median_mode(x=np.array(x[train_idx]), categorial=encoded_indices)
    
    print("imputation_values:{}".format(imputation_values))
    preprocessed = {
        'train': train,
        'test': test,
        'valid': valid,
        'end_t': end_time,
        'covariates': covariates_new,
        'one_hot_indices': encoded_indices,
        'imputation_values': imputation_values
    }
    return preprocessed


if __name__ == '__main__':
    generate_data()
