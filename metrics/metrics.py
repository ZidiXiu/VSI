import numpy as np

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