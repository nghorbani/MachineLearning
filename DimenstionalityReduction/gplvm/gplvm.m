% function [X,l,sigmaf] = gplvm(Y, q)
%warning('off', 'MATLAB:nearlySingularMatrix'); %turn off sigular matrix warning
close all; clear; clc
rng(100)

[Y, labels] = generate_data('helix', 100);
[n,d] = size(Y); q = 2;

% LogLikeandGrad2 = @(p,Y) [LogLike(p, Y), LogLikeGrad(p, Y)];
X0 = mvnrnd(zeros(n,q),eye(q,q));
X0 = PCA(Y,q);

l0 = .8;sigmaf0=0.75;        
p0 = [l0;sigmaf0];%[X0(:); l0;sigmaf0];
% [p, scgoptions, ~, ~, ~] = scg('LogLike', p0', [], 'LogLikeGrad',Y);

%p = minimize(p0, 'LogLikeandGrad', -500, Y);
diffgrads = checkgrad('LogLikeandGrad', p0, 1e-10, Y,X0)

% X = p(1:end-2);
% l = p(end-1:end-1);
% sigmaf = p(end:end);
% 
% X = reshape(X, [n q]);   
% warning('on', 'MATLAB:nearlySingularMatrix')
