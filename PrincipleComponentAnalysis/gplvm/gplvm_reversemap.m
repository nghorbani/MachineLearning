function [Ygplvmreverse,lgplvmreverse,sigmafgplvmreverse] = gplvm_reversemap(Xgplvm,Ygplvmreverse0,d)
    warning('off', 'MATLAB:nearlySingularMatrix'); %turn off sigular matrix warning
    [n,~] = size(Xgplvm);
    %Y0 = mvnrnd(zeros(n,d),eye(d));
    Y0 = Ygplvmreverse0;
    l0 = .8;sigmaf0=0.75;        
    p0 = [Y0(:); l0;sigmaf0]';
    
    scgoptions = [1,  1e-4, 1e-4, 1e-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 0, 1e-8, 0.1, 0];

    [p, ~, ~, ~, ~] = scg('LogLike', p0, scgoptions, 'LogLikeGrad',Xgplvm);
    Ygplvmreverse = p(1:end-2);
    lgplvmreverse = p(end-1:end-1);
    sigmafgplvmreverse = p(end:end);
    
    Ygplvmreverse = reshape(Ygplvmreverse, [n d]);   
    warning('on', 'MATLAB:nearlySingularMatrix')