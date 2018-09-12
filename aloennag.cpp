#include <RcppArmadillo.h>
#include <math.h>
#include <iostream>

using namespace Rcpp;
using namespace arma;
using namespace std;

//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

//[[Rcpp::export]]
/*iterative matrix inverse computation algorithm without stochastic gradient*/
mat aloennag(const vec &y, const mat &X, const mat &beta, const vec &lambda, const double &alpha, const int &max_iter, const double &thresh, const double &rate, const double &momemtum){
  int n = X.n_rows;
  int k = beta.n_cols;
  mat yalo(n,k);
  vec lambda2 = n*((1-alpha)/2)*lambda;
  double temp_lambda = lambda2(0); 
  /*cout << "Calculating ALO for lambda = " << temp_lambda << endl;*/
  vec temp_beta = beta.col(0);
  uvec E = find(abs(temp_beta)>=1e-8);
  mat X_add = X.cols(E);
  mat A = (1/(2*temp_lambda))*X_add*X_add.t()+eye<mat>(n,n);
  mat J = A.i();
  vec u = y - X*temp_beta;
  for (int i = 0; i < k; i++){
    if (i == 0){
      yalo.col(0) = y - u/diagvec(J);
    }
    else{
      vec old_beta = temp_beta;
      temp_beta = beta.col(i);
      u = y - X*temp_beta;
      double old_lambda = temp_lambda;
      temp_lambda = lambda2(i);
      /*cout << "Calculating ALO for lambda = " << temp_lambda << endl;*/
      uvec add = find((abs(temp_beta)>=1e-8)&&(abs(old_beta)<1e-8));
      uvec del = find((abs(temp_beta)<1e-8)&&(abs(old_beta)>=1e-8));
      mat X_add = X.cols(add);
      mat X_del = X.cols(del);
      A = eye<mat>(n,n)+(1/(2*temp_lambda))*((2*old_lambda)*(A-eye<mat>(n,n))+X_add*X_add.t()-X_del*X_del.t());
      int it_time = 1;
      mat Y = J;
      mat Y_new = J + rate*(eye<mat>(n,n)-A*J);
      double gradnorm = max(max(abs(Y_new-Y)));
      J = Y_new + momemtum*(Y_new-Y);
      Y = Y_new;
      /*do{
        it_time += 1;
        if (it_time >= max_iter){
          break;
        }
        else{
          Y_new = J + rate*(eye<mat>(n,n)-A*J);
          J = Y_new + momemtum*(Y_new-Y);
          gradnorm = max(max(abs(Y_new-Y)));
          Y = Y_new;
        }
        
      } while (gradnorm >= thresh);*/
      while (gradnorm>=thresh){
        it_time += 1;
        if (it_time >= max_iter){
          break;
        }
        else{
          Y_new = J + rate*(eye<mat>(n,n)-A*J);
          J = Y_new + momemtum*(Y_new-Y);
          gradnorm = max(max(abs(Y_new-Y)));
          Y = Y_new;
        }
      }
      /*cout << "The total iteration times are: " << it_time << endl;*/
      yalo.col(i) = y-u/diagvec(J);
      
    }
  }
  return yalo;
}

/*iterative matrix inverse computation algorithm with stochastic gradient with momemtum while sampling one row per time*/
//[[Rcpp::export]]
mat aloennagsto(const vec &y, const mat &X, const mat &beta, const vec &lambda, const double &alpha, const int &max_iter, const double &thresh, const double &rate, const double &momemtum){
  int n = X.n_rows;
  int k = beta.n_cols;
  mat yalo(n,k);
  vec lambda2 = n*((1-alpha)/2)*lambda;
  double temp_lambda = lambda2(0); 
  cout << "Calculating ALO for lambda = " << temp_lambda << endl;
  vec temp_beta = beta.col(0);
  uvec E = find(abs(temp_beta)!=0);
  mat X_add = X.cols(E);
  mat A = (1/(2*temp_lambda))*X_add*X_add.t()+eye<mat>(n,n);
  mat J = A.i();
  vec u = y - X*temp_beta;
  for (int i = 0; i < k; i++){
    if (i == 0){
      yalo.col(0) = y - u/diagvec(J);
    }
    else{
      temp_beta = beta.col(i);
      u = y - X*temp_beta;
      temp_lambda = lambda2(i);
      uvec E = find(abs(temp_beta)!=0);
      mat Xe = X.cols(E);
      int p = Xe.n_cols;
      int j = floor(p*randu());
      vec x_samp = Xe.col(j);
      int it_time = 1;
      mat Y = J;
      mat x_int = x_samp.t()*J;
      mat Y_new = J + rate*(eye<mat>(n,n)-J-(p/(2*temp_lambda))*x_samp*x_int);
      double gradnorm = max(max(abs(Y_new-Y)));
      J = Y_new + momemtum*(Y_new-Y);
      Y = Y_new;
      while (gradnorm>=thresh){
        it_time += 1;
        if (it_time >= max_iter){
          break;
        }
        else{
          j = floor(p*randu());
          x_samp = Xe.col(j);
          x_int = x_samp.t()*J;
          Y_new = J + rate*(eye<mat>(n,n)-J-(p/(2*temp_lambda))*x_samp*x_int);
          J = Y_new + momemtum*(Y_new-Y);
          Y = Y_new;
        }
      }
      yalo.col(i) = y-u/diagvec(J);
      
    }
  }
  return yalo;
}

/*iterative matrix inverse computation algorithm with stochastic gradient with momemtum while sampling a mini-batch per time*/
//[[Rcpp::export]]
mat aloennagstograd(const vec &y, const mat &X, const mat &beta, const vec &lambda, const double &alpha, const int &max_iter, const double &batch_per, const double &thresh, const double &rate, const double &momemtum){
  int n = X.n_rows;
  int k = beta.n_cols;
  mat yalo(n,k);
  vec lambda2 = n*((1-alpha)/2)*lambda;
  double temp_lambda = lambda2(0); 
  cout << "Calculating ALO for lambda = " << temp_lambda << endl;
  vec temp_beta = beta.col(0);
  uvec E = find(abs(temp_beta)>=1e-8);
  mat X_add = X.cols(E);
  mat A = (1/(2*temp_lambda))*X_add*X_add.t()+eye<mat>(n,n);
  mat J = A.i();
  vec u = y - X*temp_beta;
  for (int i = 0; i < k; i++){
    if (i == 0){
      yalo.col(0) = y - u/diagvec(J);
    }
    else{
      temp_beta = beta.col(i);
      u = y - X*temp_beta;
      temp_lambda = lambda2(i);
      uvec E = find(abs(temp_beta)>=1e-8);
      mat Xe = X.cols(E);
      int p = Xe.n_cols;
      int batchsize = ceil(p*batch_per);
      vec v = linspace<vec>(0, (p-1),p);
      vec samp = shuffle(v);
      uvec samp_position = find(v<ceil(p*batch_per));
      mat x_samp = Xe.cols(samp_position);
      int it_time = 1;
      mat Y = J;
      mat x_int = x_samp.t()*J;
      mat Y_new = J + rate*(eye<mat>(n,n)-J-((p/batchsize)/(2*temp_lambda))*x_samp*x_int);
      double gradnorm = max(max(abs(Y_new-Y)));
      J = Y_new + momemtum*(Y_new-Y);
      Y = Y_new;
      while (gradnorm>=thresh){
        it_time += 1;
        if (it_time >= max_iter){
          break;
        }
        else{
          vec samp = shuffle(v);
          uvec samp_position = find(v<ceil(p*batch_per));
          mat x_samp = Xe.cols(samp_position);
          Y_new = J + rate*(eye<mat>(n,n)-J-((p/batchsize)/(2*temp_lambda))*x_samp*x_int);
          J = Y_new + momemtum*(Y_new-Y);
          Y = Y_new;
        }
      }
      yalo.col(i) = y-u/diagvec(J);
      
    }
  }
  return yalo;
}

/*iterative matrix inverse computation algorithm with stochastic gradient with momemtum while sampling a mini-batch per time /w correction for no active sets detected*/
//[[Rcpp::export]]
mat aloennagstograd2(const vec &y, const mat &X, const mat &beta, const vec &lambda, const double &alpha, const int &max_iter, const double &batch_per, const double &thresh, const double &rate, const double &momemtum){
  int n = X.n_rows;
  int k = beta.n_cols;
  mat yalo(n,k);
  vec lambda2 = n*((1-alpha)/2)*lambda;
  double temp_lambda = lambda2(0); 
  vec temp_beta = beta.col(0);
  uvec E = find(abs(temp_beta)>=1e-8);
  mat X_add = X.cols(E);
  mat A = (1/(2*temp_lambda))*X_add*X_add.t()+eye<mat>(n,n);
  mat J = A.i();
  vec u = y - X*temp_beta;
  for (int i = 0; i < k; i++){
    if (i == 0){
      yalo.col(0) = y - u/diagvec(J);
    }
    else{
      double old_lambda = temp_lambda;
      vec old_beta = temp_beta;
      temp_beta = beta.col(i);
      u = y - X*temp_beta;
      temp_lambda = lambda2(i);
      uvec E = find(abs(temp_beta)>=1e-8);
      mat Xe = X.cols(E);
      int p = Xe.n_cols;
      vec v = linspace<vec>(0, (p-1),p);
      uvec add = find((abs(temp_beta)>=1e-8)&&(abs(old_beta)<=1e-8));
      int it_time;
      int batchsize;
      double gradnorm;
      mat Y_new(n,n);
      mat Y (n,n);
      if (add.n_elem == 0){
        J = (temp_lambda/old_lambda)*J;
        it_time = 0;
        batchsize = ceil(p*batch_per);
        gradnorm = 2*thresh;
      }
      else{
        mat X_add = X.cols(add);
        batchsize = ceil(p*batch_per);
        it_time = 1;
        mat Y = (temp_lambda/old_lambda)*J;
        mat x_int = X_add.t()*J;
        Y_new = Y + rate*(eye<mat>(n,n)-Y-((p/batchsize)/(2*temp_lambda))*X_add*x_int);
        gradnorm = max(max(abs(Y_new-Y)));
        J = Y_new + momemtum*(Y_new-Y);
        Y = Y_new;
      }
      while (gradnorm>=thresh){
        it_time += 1;
        if (it_time >= max_iter){
          break;
        }
        else{
          vec samp = shuffle(v);
          uvec samp_position = find(v<ceil(p*batch_per));
          mat x_samp = Xe.cols(samp_position);
          mat x_int = x_samp.t()*J;
          Y_new = J + rate*(eye<mat>(n,n)-J-((p/batchsize)/(2*temp_lambda))*x_samp*x_int);
          J = Y_new + momemtum*(Y_new-Y);
          Y = Y_new;
        }
      }
      yalo.col(i) = y-u/diagvec(J);
      
    }
  }
  return yalo;
}

/*iterative matrix inverse computation algorithm with stochastic gradient without momemtum while sampling a mini-batch per time*/
//[[Rcpp::export]]
mat aloensto(const vec &y, const mat &X, const mat &beta, const vec &lambda, const double &alpha, const int &max_iter, const double &batch_per, const double &thresh, const double &rate){
  int n = X.n_rows;
  int k = beta.n_cols;
  mat yalo(n,k);
  vec lambda2 = n*((1-alpha)/2)*lambda;
  double temp_lambda = lambda2(0); 
  cout << "Calculating ALO for lambda = " << temp_lambda << endl;
  vec temp_beta = beta.col(0);
  uvec E = find(abs(temp_beta)>=1e-8);
  mat X_add = X.cols(E);
  mat A = (1/(2*temp_lambda))*X_add*X_add.t()+eye<mat>(n,n);
  mat J = A.i();
  vec u = y - X*temp_beta;
  for (int i = 0; i < k; i++){
    if (i == 0){
      yalo.col(0) = y - u/diagvec(J);
    }
    else{
      temp_beta = beta.col(i);
      u = y - X*temp_beta;
      temp_lambda = lambda2(i);
      uvec E = find(abs(temp_beta)>=1e-8);
      mat Xe = X.cols(E);
      int p = Xe.n_cols;
      vec v = linspace<vec>(0, (p-1),p);
      vec samp = shuffle(v);
      uvec samp_position = find(v<ceil(p*batch_per));
      mat x_samp = Xe.cols(samp_position);
      int it_time = 1;
      mat x_int = x_samp.t()*J;
      mat J_new = J + rate*(eye<mat>(n,n)-p*batch_per*x_samp*x_int);
      double gradnorm = max(max(abs(J_new-J)));
      J = J_new;
      while (gradnorm>=thresh){
        it_time += 1;
        if (it_time >= max_iter){
          break;
        }
        else{
          vec samp = shuffle(v);
          uvec samp_position = find(v<ceil(p*batch_per));
          mat x_samp = Xe.cols(samp_position);
          J_new = J + rate*(eye<mat>(n,n)-p*batch_per*x_samp*x_int);
          gradnorm = max(max(abs(J_new-J)));
          J = J_new;
        }
      }
      yalo.col(i) = y-u/diagvec(J);
      
    }
  }
  return yalo;
}

//[[Rcpp::export]]
/*iterative matrix inverse computation algorithm with adagrad stochastic gradient while sampling one sample per time*/
mat aloenadagrad(const vec &y, const mat &X, const mat &beta, const vec &lambda, const double &alpha, const int &max_iter, const double &thresh, const double &epsilon, const double &delta){
  int n = X.n_rows;
  int k = beta.n_cols;
  mat yalo(n,k);
  vec lambda2 = n*((1-alpha)/2)*lambda;
  double temp_lambda = lambda2(0); 
  cout << "Calculating ALO for lambda = " << temp_lambda << endl;
  vec temp_beta = beta.col(0);
  uvec E = find(abs(temp_beta)>=1e-8);
  mat X_add = X.cols(E);
  mat A = (1/(2*temp_lambda))*X_add*X_add.t()+eye<mat>(n,n);
  mat J = A.i();
  vec u = y - X*temp_beta;
  for (int i = 0; i < k; i++){
    if (i == 0){
      yalo.col(0) = y - u/diagvec(J);
    }
    else{
      vec old_beta = temp_beta;
      temp_beta = beta.col(i);
      u = y - X*temp_beta;
      double old_lambda = temp_lambda;
      temp_lambda = lambda2(i);
      cout << "Calculating ALO for lambda = " << temp_lambda << endl;
      uvec add = find((abs(temp_beta)>=1e-8)&&(abs(old_beta)<1e-8));
      uvec del = find((abs(temp_beta)<1e-8)&&(abs(old_beta)>=1e-8));
      mat X_add = X.cols(add);
      mat X_del = X.cols(del);
      A = eye<mat>(n,n)+(1/(2*temp_lambda))*((2*old_lambda)*(A-eye<mat>(n,n))+X_add*X_add.t()-X_del*X_del.t());
      int it_time = 1;
      mat Y = A*J-eye<mat>(n,n);
      double gradnorm = max(max(abs(Y)));
      mat r = pow(Y,2);
      J = J - (epsilon/(delta+pow(r,0.5)))%Y;
      do{
        it_time += 1;
        if (it_time >= max_iter){
          break;
        }
        else{
          Y = A*J-eye<mat>(n,n);
          gradnorm = max(max(abs(Y)));
          r = r+sum(sum(pow(Y,2)));
          J = J - (epsilon/(delta+pow(r,0.5)))%Y;
        }
        
      } while (gradnorm >= thresh);
      cout << "The total iteration times are: " << it_time << endl;
      yalo.col(i) = y-u/diagvec(J);
      
    }
  }
  return yalo;
}

//[[Rcpp::export]]
/*Adjusting the learning rate and momemtum according to the frobenius norm of the original matrix without stochastic gradient*/
mat aloenadnag(const vec &y, const mat &X, const mat &beta, const vec &lambda, const double &alpha, const int &max_iter, const double &thresh, const double &rate, const double &momemtum){
  int n = X.n_rows;
  int k = beta.n_cols;
  mat yalo(n,k);
  vec lambda2 = n*((1-alpha)/2)*lambda;
  double temp_lambda = lambda2(0); 
  /*cout << "Calculating ALO for lambda = " << temp_lambda << endl;*/
  vec temp_beta = beta.col(0);
  uvec E = find(abs(temp_beta)>=1e-8);
  mat X_add = X.cols(E);
  int active = X_add.n_cols;
  mat A = (2*temp_lambda)*eye<mat>(active,active)+X_add.t()*X_add;
  mat J = eye<mat>(n,n)-X_add*A.i()*X_add.t();
  vec u = y - X*temp_beta;
  for (int i = 0; i < k; i++){
    if (i == 0){
      yalo.col(0) = y - u/diagvec(J);
    }
    else{
      temp_beta = beta.col(i);
      u = y - X*temp_beta;
      temp_lambda = lambda2(i);
      if (sum((abs(temp_beta)>=1e-8)) < floor(n*2/3)){
        uvec E = find(abs(temp_beta)>=1e-8);
        mat X_add = X.cols(E);
        int active = X_add.n_cols;
        mat A = (2*temp_lambda)*eye<mat>(active,active)+X_add.t()*X_add;
        mat J = eye<mat>(n,n)-X_add*A.i()*X_add.t();
        vec u = y - X*temp_beta;
        yalo.col(i) = y - u/diagvec(J);
        cout << "computing in primal approach" << endl;
      }
      else{
        /*cout << "Calculating ALO for lambda = " << temp_lambda << endl;*/
        uvec add = find((abs(temp_beta)>=1e-8));
        mat X_add = X.cols(add);
        mat A = eye<mat>(n,n)+(1/(2*temp_lambda))*X_add*X_add.t();
        double fro = 1/sum(sum(pow(A,2)));
        int it_time = 1;
        mat Y = J;
        mat Y_new = J + rate*fro*(eye<mat>(n,n)-A*J);
        double gradnorm = max(max(abs(Y_new-Y)));
        J = Y_new + momemtum*fro*(Y_new-Y);
        Y = Y_new;
        while (gradnorm>=thresh){
          it_time += 1;
          if (it_time >= max_iter){
            break;
          }
          else{
            Y_new = J + fro*rate*(eye<mat>(n,n)-A*J);
            J = Y_new + fro*momemtum*(Y_new-Y);
            gradnorm = max(max(abs(Y_new-Y)));
            Y = Y_new;
            }
          }
        /*cout << "The total iteration times are: " << it_time << endl;*/
        cout << "computing in dual approach" << endl;
        yalo.col(i) = y-u/diagvec(J);
      }
    }
  }
  return yalo;
}
