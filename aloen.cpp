#include <RcppArmadillo.h>
#include <math.h>

using namespace Rcpp;
using namespace arma;

//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

//[[Rcpp::export]]
vec alo_elnet(const vec &y, const mat &X,const vec &beta, const double &lambda1, const double &lambda2){
  vec u = y - X*beta; 
  vec sing = X.t()*u;
  uvec E = find(abs(sing)>lambda1);
  mat Xe = X.cols(E);
  mat Ie = eye<mat>(Xe.n_cols,Xe.n_cols);
  mat Je = (1/(1+2*lambda2))*eye<mat>(Xe.n_cols,Xe.n_cols);
  mat H = Xe*inv_sympd(Je*Xe.t()*Xe+Ie-Je)*Je*Xe.t();
  vec y_tilde = X*beta + diagvec(H)/(1-diagvec(H))%(X*beta-y);
  return y_tilde;
}

//[[Rcpp::export]]
vec aloeldual(const vec &y, const mat &X, const vec &beta, const double &lambda, const double &alpha){
  double n = X.n_rows;
  double lambda2 = n*lambda*(1-alpha)/2;
  vec u = y-X*beta;
  uvec E = find(abs(X.t()*u)>n*lambda*alpha);
  mat Xe = X.cols(E);
  mat J = inv_sympd(eye<mat>(X.n_rows,X.n_rows)+(1/(2*lambda2))*Xe*Xe.t());
  vec yalo = y - u/diagvec(J);
  return yalo;
}