import numpy as np


def simulation_cox_gompertz(n=10000, p=2, pc=0, pc_level=4, pval=[1/4.]*4, lambda_=7e-8, alpha_=0.2138, censor_bound= 68,seed=123):
    # linear relationship
    # n: number of patients
    # p: number of total covariates
    # pc: number of categorical variables
    # pc_level: levels of categorical variable
    # pval: probabilities for each level
    # lambda_exp: parameters for baseline hazards
    # censor_bound: upper bound of censoring generating process
    p_total = (p-pc) + pc*pc_level
    np.random.seed(seed)
    if p-pc == 2:
        # generate based on Bender's paper
        beta_linear = np.array([0.15,0.001])
        X_age = np.random.normal(loc=24.3, scale = 8.38, size=n).reshape((n,1))
        X_randon = np.random.normal(loc=266.84, scale = 507.82, size=n).reshape((n,1))

        X_cts = np.concatenate((X_age, X_randon), axis=1)
    else:
        beta_linear = np.random.normal(0,1,p_total)
        X_cts = [np.random.normal(loc=0, scale=1.0, size=n) for i in range(p-pc)]
        X_cts = np.transpose(np.stack(X_cts, axis=0 ))

    if (pc>0):
        X_cat = [np.random.multinomial(1, pval, size=n) for i in range(pc)]
        X_cat = np.stack(X_cat, axis=0 ).reshape(n,pc_level*pc)
        one_hot_indices = [[i,i+1,i+2,i+3] for i in np.arange(p-pc,p_total, pc_level)]    
        X = np.concatenate((X_cts, X_cat), axis=1)
    else:
        X = X_cts
        one_hot_indices = []
        
    # the problem with the left tail is about this U!
    U = np.random.uniform(size=n)
    # generate T accordingly
    T = (1/alpha_)*np.log(1-alpha_*np.log(U)/(lambda_*np.exp(np.dot(X,beta_linear))))
    
    if censor_bound>0:
        sidx = np.argsort(T,axis=0)
        TS = T[sidx]
        XS = X[sidx,:]
        np.random.seed(seed)
        # change the way censoring is defined
        # first set the maximum censoring at: censor_bound
    #    C = np.repeat(censor_bound,n)
        right_truncate = T<censor_bound
        EPS = 0
        # define C only for the right truncated samples
        C = np.random.uniform(0+EPS,censor_bound,size=len(T[right_truncate]))
        CS = np.concatenate([C,np.repeat(censor_bound,n-len(T[right_truncate]))])
        event = 1*(TS<CS)
        nonevent = CS<TS
        # observed time
        YS = TS.copy()
        YS[nonevent] = CS[nonevent]

        # shuffle back to unsorted order
        perm_idx = np.random.permutation(n)
        X = XS[perm_idx,:]
        Y = YS[perm_idx]
        event = event[perm_idx]
        C = CS[perm_idx]
        T = TS[perm_idx]
    else:
        Y = T.copy()
        C = 0
        event = np.ones(n)
    
    return({"t": Y, "e":event, "x":X, "T": T,"C":C, 'one_hot_indices':one_hot_indices})

def simulate_AFT_Lognormal(n, censor_bound, seed):
    a0 = 2
    a1 = 1
    a2 = 1
    a_ = np.array([a0, a1, a2])
    sigma_ = 1
    # epsilon_ = np.array(np.random.exponential(scale=3, size=n))
    epsilon_ = np.array(np.random.normal(size=n))
    X_1 = np.random.binomial(1,0.5, size=n).reshape((n,1))
    X_2 = np.random.normal(loc=0, scale = 1, size=n).reshape((n,1))

    X = np.concatenate((X_1, X_2), axis=1)    # X for aft
    # design matrix for simulation
    x_design = np.concatenate([np.ones([n,1]), X],axis=1)
    log_T = np.sum(np.multiply(x_design, a_),axis=1) + epsilon_*sigma_
    T_ = np.exp(log_T)

    if censor_bound >0:
        sidx = np.argsort(T_,axis=0)
        TS = T_[sidx]
        XS = X[sidx,:]
        np.random.seed(seed)
        # change the way censoring is defined
        # first set the maximum censoring at: censor_bound
        #    C = np.repeat(censor_bound,n)
        right_truncate = T_<censor_bound
        EPS = 0
        # define C only for the right truncated samples
        C = np.random.uniform(0+EPS,censor_bound,size=len(T_[right_truncate]))
        CS = np.concatenate([C,np.repeat(censor_bound,n-len(T_[right_truncate]))])
        event = 1*(TS<CS)
        nonevent = CS<TS
        # observed time
        YS = TS.copy()
        YS[nonevent] = CS[nonevent]

        # shuffle back to unsorted order
        perm_idx = np.random.permutation(n)
        X = XS[perm_idx,:]
        Y = YS[perm_idx]
        event = event[perm_idx]
        C = CS[perm_idx]
        T_ = TS[perm_idx]
        
    else:
        Y = T_.copy()
        event = np.ones(n)
        C = 0

    
    return({"t": Y, "e":event, "x":X, "T": T_,"C":C})

def simulate_AFT_Lognormal2(n, censor_bound, seed, EPS = 1e-8):
    a0 = 2
    a1 = 1
    a2 = 1
    a_ = np.array([a0, a1, a2])
    sigma_ = 1
    # epsilon_ = np.array(np.random.exponential(scale=3, size=n))
    epsilon_ = np.array(np.random.normal(size=n))
    X_1 = np.random.binomial(1,0.5, size=n).reshape((n,1))
    X_2 = np.random.normal(loc=0, scale = 1, size=n).reshape((n,1))

    X = np.concatenate((X_1, X_2), axis=1)    # X for aft
    # design matrix for simulation
    x_design = np.concatenate([np.ones([n,1]), X],axis=1)
    log_T = np.sum(np.multiply(x_design, a_),axis=1) + epsilon_*sigma_
    T_ = np.exp(log_T)
    # using different censoring strategy
    if censor_bound > 0:
        C = np.random.uniform(0+EPS,censor_bound,size=n)
        event = 1*(T_<C)
        Y = T_.copy()
        Y[C<=T_] = T_[C<=T_]

    else:
        Y = T_.copy()
        event = np.ones(n)
        C = 0

    
    return({"t": Y, "e":event, "x":X, "T": T_,"C":C})