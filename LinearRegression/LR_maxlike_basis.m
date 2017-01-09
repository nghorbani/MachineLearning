function [yPrediction] = LR_maxlike_basis(Xtrain,Ytrain,Xtest,M,basis_fun)

N = length(Xtrain);

phi = zeros(N,M); % gramian matrix
for i = 1:M
    phi(:,i) = basis_fun(Xtrain,Xtrain(i));
end

wML = pinv(phi)*Ytrain;%  wML = (phi'*phi)^-1*phi'*Ytrain;

sigmaML = 0;
for i = 1:N
    sigmaML = sigmaML + (Ytrain(i)-wML'*phi(1,:)')^2;
end
sigmaML = sqrt((1/N)* sigmaML);

predicted_phi = zeros(length(Xtest),M); % gramian matrix
for i = 1:M
    predicted_phi(:,i) = basis_fun(Xtest,Xtrain(i));
end

yPrediction = wML' * predicted_phi';
