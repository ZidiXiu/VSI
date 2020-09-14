library('RcppCNPy')
library("survival")
library('rms')

file_path = '/data/zidi/cVAE/datasets/cox_gompertz'
test_path = paste(file_path,"cox_gompertz_30_test.csv",sep='/')
test <- read.csv(file=test_path, header=TRUE)
test_pred = read.csv(file=paste(file_path,"cox_coxnet_er30_pred.csv",sep='/'))

file_path = '/data/zidi/cVAE/datasets'
name = 'flchain'
result_path = paste('/data/zidi/cVAE/results',name,sep='/')
test_path = paste(file_path,"flchain_test.csv",sep='/')
test <- read.csv(file=test_path, header=TRUE)
tt = npyLoad(file=paste(result_path,paste("cVAE_q_",name,"_tt.npy",sep=""),sep='/'))
test_pred = read.csv(file=paste(result_path,"flchain_coxnet_pred.csv",sep='/'))

file_path = '/data/zidi/cVAE/datasets'
name = 'support'
result_path = paste('/data/zidi/cVAE/results',name,sep='/')
test_path = paste(file_path,"support_test.csv",sep='/')
test <- read.csv(file=test_path, header=TRUE)
tt = npyLoad(file=paste(result_path,paste("cVAE_q_",name,"_tt.npy",sep=""),sep='/'))
test_pred = read.csv(file=paste(result_path,"support_coxnet_pred.csv",sep='/'))

file_path = '/data/zidi/cVAE/datasets'
name = 'seer'
result_path = paste('/data/zidi/cVAE/results',name,sep='/')
test_path = paste(file_path,"seer_test.csv",sep='/')
test <- read.csv(file=test_path, header=TRUE)
tt = npyLoad(file=paste(result_path,paste("cVAE_q_",name,"_tt.npy",sep=""),sep='/'))
test_pred = read.csv(file=paste(result_path,"seer_coxnet_pred.csv",sep='/'))


w <- rcorr.cens(x=-test_pred[[2]], S= Surv(test[['t']],test[['e']]))
C <- w['C Index']
se <- w['S.D.']/2

CI <- paste(round(C-1.96*se,3), round(C+1.96*se,3))
CI

