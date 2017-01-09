function [yPrediction, var_predictive ]= LR_bayes_basis(Xtrain,Ytrain,Xtest,M,basis_fun,alpha)
if nargin<6
    alpha = 1;
end
N = length(Xtrain);
beta = 1;

phi = zeros(N,M); % gramian matrix
for i = 1:M
    phi(:,i) = basis_fun(Xtrain,Xtrain(i));
end


sigma_posterior = (alpha*eye(M,M)+beta*phi'*phi)^-1;
u_posterior = beta*sigma_posterior*phi'*Ytrain;

predicted_phi = zeros(length(Xtest),M); % gramian matrix
for i=1:M
    predictive_phi(:,i) = basis_fun(Xtest,Xtrain(i));
end

predictive_mean = u_posterior'*predictive_phi';
predictive_sigma = beta + predictive_phi*sigma_posterior*predictive_phi';

var_predictive = diag(predictive_sigma);
yPrediction = predictive_mean;
end
