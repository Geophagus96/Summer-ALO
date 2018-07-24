tic;
n = 300;
p = 600;
k = 60;
method = input('Please input the simulation method:(1 for correlated; 2 for misspecification; 3 for heavytailed)');
%generating the parameters that are valid
valid = randperm(p,k);
beta = zeros(p,1);
beta(valid,:) = randn(k,1);
beta = [0.1;beta];
if method == 1
    epsilon = 0.5*randn(n,1);
    r = 0.8.^(1:p);
    c = toeplitz(r)./k;
    X = mvnrnd(zeros(p,1),c,n);
    X = [ones(n,1),X];
    y = X*beta+epsilon;
elseif method == 2
    X = randn(n,p)./sqrt(k);
    X = [ones(n,1),X];
    epsilon = 0.5*randn(n,1);
    y = X*beta+epsilon;
    y = sign(y).*sqrt(abs(y));
elseif method == 3
    X = randn(n,p)./sqrt(k);
    X = [ones(n,1),X];
    epsilon = 0.5*trnd(3,n,1);
    y = X*beta+epsilon;
end
lambda = exp(linspace(log(10^-2.5),log(10^-1.5),30));
D = [zeros(p,1),eye(p)];
risk = zeros(1,30);
for i = 1:size(lambda,2)
        l = n*lambda(1,i);
        cvx_begin quiet
          cvx_precision('high')
          cvx_solver('sedumi')
          cvx_solver_settings('max_iter',1000)
          variable theta(n) 
          variable u(p)
          minimize norm((y-theta),2)
          subject to
            u <= l
            -u <= l
            X'*theta == D'*u
        cvx_end
        E = abs(abs(theta'*X)-l)<1e-8;
        Xe = [ones(n,1),X(:,E)];
        H = eye(n)-Xe*inv(Xe'*Xe)*Xe';
        yalo = y-theta./diag(H);
        risk(1,i) = mean((y-yalo).^2);
end
t = toc
if method == 1
    save('F:\\alohighdim\\alo_cor.mat','X','y','risk');
elseif method == 2
    save('F:\\alohighdim\\alo_mis.mat','X','y','risk');
elseif method == 3
    save('F:\\alohighdim\\alo_hvy.mat','X','y','risk');
end
plot(log(lambda),risk,'*--');
