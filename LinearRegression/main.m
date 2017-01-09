% Nima Ghorbani
% Goal: Standard Linear Regression Model With Gaussian Noise
% Project to a high dimensional feature space and then using the LR model
% Bayesian and Maximum Likelihood

clear; clc; close all;

set(0,'DefaultFigureWindowStyle','docked')
load('data.mat')

N = length(Xtrain);
%% In high dimensional feature space - basis functions

M = N; % using a 50 dimentional basis
sigma_phis = [.1 1 10]; % width parameter

basis_fun = @(x,j,s_phi) exp(-(0.5*(x-j).^2)/(s_phi^2));
for alpha = [1, 10^-8]
    figure();i=0;
    for sigma_phi = sigma_phis
        i=i+1;subplot(2,2,i);
        ML_yPrediction = LR_maxlike_basis(Xtrain, Ytrain, Xtest, M, @(x,j) basis_fun(x,j,sigma_phi));
        [bayes_yPrediction, var_Prediction ]= LR_bayes_basis(Xtrain,Ytrain,Xtest,M,@(x,j) basis_fun(x,j,sigma_phi),alpha);
        plot(Xtest,Ytest,'g');hold on;
        plot(Xtest,ML_yPrediction,'r');
        errorbar(Xtest,bayes_yPrediction,var_Prediction,'m')
        xlim([0,7]);ylim([-3,3]);
        title(sprintf('\\sigma=%1.1f, \\alpha=%1.0e',sigma_phi,alpha));
        legend('original','MLE','bayes','Location','southwest')
    end
    suptitle(sprintf('LR in Feature Space\nDifferent basis function parameters'));
end
