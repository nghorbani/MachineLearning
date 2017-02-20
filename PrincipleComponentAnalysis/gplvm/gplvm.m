function [X,l,sigmaf] = gplvm(Y, q)
    warning('off', 'MATLAB:nearlySingularMatrix'); %turn off sigular matrix warning
    [n,~] = size(Y);

%     [U, ~, ~] = svd(cov(Y));
% 
%     Um = U(:,1:q);
%     X0 = Y*Um;%initializing with PCA 
    X0 = mvnrnd(zeros(n,q),eye(q,q));
    l0 = .8;sigmaf0=0.75;        
     p0 = [X0(:); l0;sigmaf0]';
    
    scgoptions = [1,  1e-10, 1e-10, 1e-10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 0, 0, 0, 0];
%     p = minimize(p0', 'LogLikeandGrad', -500, Y);
    [p, scgoptions, ~, ~, ~] = scg('LogLike', p0, scgoptions, 'LogLikeGrad',Y);
%     diffgrads = checkgrad('LogLikeandGrad', p0', 1e-10, Y)
    %scgoptions(10) % number of iterations 
    X = p(1:end-2);
    l = p(end-1:end-1);
    sigmaf = p(end:end);
    
    X = reshape(X, [n q]);   
    warning('on', 'MATLAB:nearlySingularMatrix')