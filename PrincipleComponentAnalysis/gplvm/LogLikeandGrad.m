function [L, dLdP] = LogLikeandGrad(p, Y)

    X = p(1:end-2);
    l = p(end-1:end-1);
    sigmaf = p(end:end);

    [n, d] = size(Y);
    q = ceil(numel(X) ./ n);
    X = reshape(X, [n q]);

    sum_X = sum(X .^ 2, 2);
    sqDistance = bsxfun(@plus, bsxfun(@plus, -2 * (X * X'), sum_X), sum_X');
    K = (sigmaf^2)*exp((-l*0.5)*sqDistance);
    
    Kinv = pinv(K);
    KinvYYT = (Kinv * Y) * Y';
    dLdK = KinvYYT * Kinv - d * Kinv;

%     dLdX = zeros(n, q);
%     for i=1:n
%         dLdX(i,:) = sum(-l*dLdK(i,:)'.*(K(i,:)'.*(X(i,:)-X)), 1);
%     end
        
    dLdX = zeros(n, q);
    for i=1:q
        K1 = reshape(X(:,i),n,1);
        K2 = reshape(X(:,i),1,n);
        dKdX = l.*(K1-K2).*K;
        dLdX(:,i) = (sum(dLdK.*dKdX,1));
    end
 
    dLdl = 0.5*sum(sum(dLdK.*(-0.5*sqDistance.*K)));%0.5*trace((KinvYYT-Kinv)*(l^-3)*K);
    dLdsigmaf = sum(sum(dLdK.*(K / sigmaf)));
    dLdP = [dLdX(:); dLdl; dLdsigmaf];

    L = -((d * n) / 2) * log(2 * pi) - (d / 2) * log(det(K) + realmin) - .5 * trace(KinvYYT);