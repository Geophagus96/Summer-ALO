setwd('F:\\alohighdim')
library(R.matlab)
library(glmnet)
library(ggplot2)
method = readline('Please input simulation method (1 for correlated; 2 for misspecification; 3 for heavytailed):')
if (method == '1'){
x_int_cor = readMat('alo_cor.mat')
X_int = x_int_cor$X
X_int = X_int[,-1]
n = nrow(X_int)
y_int = x_int_cor$y
risk_alo = x_int_cor$risk
lambda = exp(seq(from = log(10^-3),to = log(10^-1.5),length = 30))
result=cv.glmnet(X_int,y_int,lambda = lambda,family="gaussian",nfolds = n,type.measure = 'mse',thresh = 1e-14,maxit = 10000, grouped = FALSE, intercept = TRUE,standardize = FALSE)
}else if (method == '2'){
  x_int_mis = readMat('alo_mis.mat')
  X_int = x_int_mis$X
  X_int = X_int[,-1]
  n = nrow(X_int)
  y_int = x_int_mis$y
  risk_alo = x_int_mis$risk
  lambda = exp(seq(from = log(10^-3),to = log(10^-1.5),length = 30))
  result=cv.glmnet(X_int,y_int,lambda = lambda,family="gaussian",nfolds = n,type.measure = 'mse',thresh = 1e-14,maxit = 10000, grouped = FALSE, intercept = TRUE,standardize = FALSE)
}else if (method == '3'){
  x_int_hvy = readMat('alo_hvy.mat')
  X_int = x_int_hvy$X
  X_int = X_int[,-1]
  n = nrow(X_int)
  y_int = x_int_hvy$y
  risk_alo = x_int_hvy$risk
  lambda = exp(seq(from = log(10^-2.5),to = log(10^-1.5),length = 30))
  result=cv.glmnet(X_int,y_int,lambda = lambda,family="gaussian",nfolds = n,type.measure = 'mse',thresh = 1e-10,maxit = 10000, grouped = FALSE, intercept = TRUE,standardize = FALSE)
}
ggplot()+
  geom_line(aes(x = lambda, y = as.vector(risk_alo)),col='blue')+
  geom_point(aes(x= lambda, y=as.vector(risk_alo)),pch = 5,col= 'blue')+
  geom_line(aes(x = result$lambda, y = result$cvm),col='black')+
  geom_point(aes(x=result$lambda,y = result$cvm),pch = 5,col= 'black')+
  xlab('lambda')+
  ylab('risk')


