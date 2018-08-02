#include <RcppArmadillo.h>
#include <math.h>
#include <iostream>
using namespace Rcpp;
using namespace arma;
using namespace std;
//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

vec lognetALOpr(const vec &beta, const mat &X, const vec &y, const double &lambda, const double &alpha);
vec lognetALOdl(const vec &beta, const mat &X, const vec &y, const double &lambda, const double &alpha);
//[[Rcpp::export]]
vec lognetALO(const vec &beta, const mat &X, const vec &y, const double &lambda, const double &alpha, const int &method){
  switch(method)
  {
    case (1):
      {
         cout << "Computing ALO in the primal approach" << endl;
         return lognetALOpr(beta, X, y, lambda, alpha);
         break;
      }
    case (2):
      {
        cout << "Computing ALO in the dual approach" << endl;
        return lognetALOpr(beta, X, y, lambda, alpha);
        break;
      }
    default:
      {
        cout << "Not a valid approach" << endl;
      }
  }
}

vec lognetALOdl(const vec &beta, const mat &X, const vec &y, const double &lambda, const double &alpha){
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

vec lognetALOpr(const vec &beta, const mat &X, const vec &y, const double &lambda, const double &alpha) {
  vec yhat = X * beta;
  vec eXb = exp(yhat);
  uvec E = find(abs(beta) >= 1e-8);
  mat XE = X.cols(E);
  mat hessR = (1 - alpha) * lambda * eye<mat>(E.n_elem, E.n_elem);
  mat H = XE * inv_sympd(XE.t() * diagmat(eXb / square(1.0 + eXb)) * XE / X.n_rows + hessR) * XE.t();
  vec y_alo = yhat + diagvec(H) % (eXb / (1.0 + eXb) - y) / (X.n_rows - diagvec(H) % (eXb / square(1.0 + eXb)));
  
  return y_alo;
}