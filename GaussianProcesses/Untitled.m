load mnist_train
Xtrain0 = train{10}; Xtrain1 = train{1};
Xtrain2 = [Xtrain0,Xtrain1]'; % X = N*D
X2 = Xtrain2(1:100,:);
X1 = Xtrain2(1:500,:);

%%

sigman = 0; % noise of the reading
sigmaf = 1; % parameters of the GP - next to be computed by optimization
l=0.2;

K = get_kernel(X1, X2, sigmaf, l, 0);
K2 = get_kernel2(X1, X2, sigmaf, l, 0);
sum(sum(K==K2))
function K = get_kernel(X1,X2,sigmaf,l,sigman)
    sum_X1 = sum(X1 .^ 2, 2);
    sum_X2 = sum(X2 .^ 2, 2);
    K = (sigmaf^2)*exp(bsxfun(@plus, bsxfun(@plus, -2 * (X1 * X2'), sum_X1), sum_X2') / (-2 * l ^ 2));
    if size(K,1)== size(K,2)
        K = K+(sigman^2)*eye(size(X1,1));
    end
end

function K = get_kernel2(x1,x2,sigmaf,l,sigman)
    k = @(x1,x2,sigmaf,l,sigman) (sigmaf^2)*exp(-(1/(2*l^2))*(x1-x2)*(x1-x2)') + (sigman^2);
    K = zeros(size(x1,1),size(x2,1));
    for i = 1:size(x1,1)
        for j = 1:size(x2,1)
            if i==j;K(i,j) = k(x1(i,:),x2(j,:),sigmaf,l,sigman);
            else;K(i,j) = k(x1(i,:),x2(j,:),sigmaf,l,0);end
        end
    end
end