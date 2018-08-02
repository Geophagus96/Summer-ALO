#include <RcppArmadillo.h>
#include <math.h>
using namespace Rcpp;
using namespace arma;
using namespace std;
//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]


//[[Rcpp::export]]
vec loglassoALO(const vec &beta, const mat &X, const vec &y, const double &lambda){
  double n = X.n_rows;
  double lambda1 = n*lambda;
  vec theta = y - exp(X*beta)/(1+exp(X*beta));
  vec yhat = X*beta;
  vec l2 = 1/((y-theta)%(1-y+theta));
  vec K = sqrt(l2);
  vec yu = (theta%l2+yhat)/K;
  mat Xu = diagmat((1/K))*X;
  uvec E = find(abs(X.t()*theta)>lambda1);
  mat Xue = Xu.cols(E);
  mat J = eye<mat>(X.n_rows,X.n_rows) - Xue*inv_sympd(Xue.t()*Xue)*Xue.t();
  vec yalo = K%(yu - K%theta/diagvec(J));
  return yalo;
}


//[[Rcpp::export]]
vec logelnetALO(const vec &beta, const mat &X, const vec &y, const double &lambda, const double &alpha){
  double n = X.n_rows;
  double lambda1 = n*lambda*alpha;
  double lambda2 = n*lambda*(1-alpha)/2;
  vec theta = y - exp(X*beta)/(1+exp(X*beta));
  vec yhat = X*beta;
  vec l2 = 1/((y-theta)%(1-y+theta));
  vec K = sqrt(l2);
  vec yu = (theta%l2+yhat)/K;
  mat Xu = diagmat((1/K))*X;
  uvec E = find(abs(X.t()*theta)>lambda1);
  mat Xue = Xu.cols(E);
  mat J = inv_sympd(eye<mat>(X.n_rows,X.n_rows)+(1/(2*lambda2))*Xue*Xue.t());
  vec yalo = K%(yu - K%theta/diagvec(J));
  return yalo;
}