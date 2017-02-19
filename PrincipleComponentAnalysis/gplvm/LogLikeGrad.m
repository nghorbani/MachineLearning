function dLdP = LogLikeGrad(p, Y)
    
    X = p(1:end-2);
    l = p(end-1:end-1);
    sigmaf = p(end:end);

    [n, d] = size(Y);
    q = ceil(numel(X) ./ n);
    X = reshape(X, [n q]);

    % kernel computations
    sum_X = sum(X .^ 2, 2);
    sqDistance = bsxfun(@plus, bsxfun(@plus, -2 * (X * X'), sum_X), sum_X');
    K = (sigmaf^2)*exp((-l*0.5)*sqDistance);
    
    % gradients computations
    %   dLdK = invK * Y * Y' * invK - d * invK;
    %Kinv = pinv(K);
    %KinvYYT = (Kinv * Y) * Y';
    %dLdK = KinvYYT * Kinv - d * Kinv;
    dLdK = ((K\Y)*Y'  - d*eye(n))/ K;


    dLdX = zeros(n, q);
    for i=1:n
        dLdX(i,:) = sum(-l*dLdK(i,:)'.*(K(i,:)'.*(X(i,:)-X)), 1);
    end
        
%     % Compute gradient with respect to coordinates
%     dLdX = zeros(n, q);
%     for i=1:q
%         K1 = reshape(X(:,i),n,1);
%         K2 = reshape(X(:,i),1,n);
%         dKdX = -l*(K1-K2).*K;
%         dLdX(:,i) = (2*sum(dLdK.*dKdX,1));
%     end
 
    dLdl = 0.5*sum(sum(dLdK.*(-0.5*sqDistance.*K)));%0.5*trace((KinvYYT-Kinv)*(l^-3)*K);
    dLdsigmaf = sum(sum(dLdK.*(K / sigmaf)));
    dLdP = [dLdX(:); dLdl; dLdsigmaf]';