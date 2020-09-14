library('RcppCNPy')
library("survival")
library('rms')

file_path = '/data/zidi/cVAE/datasets/cox_gompertz'
result_path = '/data/zidi/cVAE/results/coxph'
name = 'er30'
test_path = paste(file_path,"cox_gompertz_30_test.csv",sep='/')

# file_path = '/data/zidi/cVAE/datasets'
# name = 'seer'
# result_path = paste('/data/zidi/cVAE/results',name,sep='/')
# test_path = paste(file_path,"seer_test.csv",sep='/')

test <- read.csv(file=test_path, header=TRUE)
# tt = npyLoad(file=paste(result_path,paste("cVAE_q_",name,"_tt.npy",sep=""),sep='/'))


# for VSI
filename = paste('cVAE_q_',name,"_test_pred_avgt.npy",sep="")
test_pred = npyLoad(file=paste(result_path,filename,sep='/'))
w <- rcorr.cens(x=test_pred, S= Surv(test[['t']],test[['e']]))
C <- w['C Index']
se <- w['S.D.']/2
CI <- paste(round(C,4), round(C-1.96*se,4), round(C+1.96*se,4))
CI

filename = paste('cVAE_q_',name,"_test_pred_medt.npy",sep="")
test_pred = npyLoad(file=paste(result_path,filename,sep='/'))
w <- rcorr.cens(x=test_pred, S= Surv(test[['t']],test[['e']]))
C <- w['C Index']
se <- w['S.D.']/2

CI <- paste(round(C,4), round(C-1.96*se,4), round(C+1.96*se,4))
CI

# for VSI-noq
# test_pred_prob = npyLoad(file=paste(result_path,"cVAE_noq_er100_test_pred_prob.npy",sep='/'))
# mySum = abs(t(apply(test_pred_prob, 1, cumsum)-0.5))
# medt_idx = apply(mySum, 1, which.min)
filename = paste('cVAE_noq_',name,"_test_pred_medt.npy",sep="")
test_pred = npyLoad(file=paste(result_path,filename,sep='/'))
w <- rcorr.cens(x=test_pred, S= Surv(test[['t']],test[['e']]))
C <- w['C Index']
se <- w['S.D.']/2

CI <- paste(round(C,4), round(C-1.96*se,4), round(C+1.96*se,4))
CI


# for MLP
filename = paste('MLP_',name,"_test_pred_medt.npy",sep="")
test_pred = npyLoad(file=paste(result_path,filename,sep='/'))
w <- rcorr.cens(x=test_pred, S= Surv(test[['t']],test[['e']]))
C <- w['C Index']
se <- w['S.D.']/2

CI <- paste(round(C,4), round(C-1.96*se,3), round(C+1.96*se,3))
CI

# for RSF
filename = paste('MLP_',name,"_test_pred_medt.npy",sep="")
test_pred = npyLoad(file=paste(result_path,filename,sep='/'))
w <- rcorr.cens(x=test_pred, S= Surv(test[['t']],test[['e']]))
C <- w['C Index']
se <- w['S.D.']/2

CI <- paste(round(C,4), round(C-1.96*se,3), round(C+1.96*se,3))
CI

# for AFT-Weibull
filename = paste('AFT_',name,"_pred_t.npy",sep="")
test_pred = npyLoad(file=paste(result_path,filename,sep='/'))
w <- rcorr.cens(x=test_pred, S= Surv(test[['t']],test[['e']]))
C <- w['C Index']
se <- w['S.D.']/2

CI <- paste(round(C,4), round(C-1.96*se,4), round(C+1.96*se,4))
CI

# for Cox
filename = paste('cox_',name,"_pred_h.npy",sep="")
test_pred = npyLoad(file=paste(result_path,filename,sep='/'))
w <- rcorr.cens(x=-test_pred, S= Surv(test[['t']],test[['e']]))
C <- w['C Index']
se <- w['S.D.']/2

CI <- paste(round(C,4), round(C-1.96*se,3), round(C+1.96*se,3))
CI