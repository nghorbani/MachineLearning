function L = LogLike(p, Y)

    X = p(1:end-2);
    l = p(end-1:end-1);
    sigmaf = p(end:end);

    [n, d] = size(Y);
    q = ceil(numel(X) ./ n);
    X = reshape(X, [n q]);

    sum_X = sum(X .^ 2, 2);
    sqDistance = bsxfun(@plus, bsxfun(@plus, -2 * (X * X'), sum_X), sum_X');
    K = (sigmaf^2)*exp((-l*0.5)*sqDistance);
    
    %   dLdK = invK * X * X' * invK - d * invK;
    %Kinv = pinv(K);
    %KinvYYT = (Kinv \ Y) * Y';
    KinvYYT = (K \ Y) * Y';

    % Compute log-likelihood
    L = -((d * n) / 2) * log(2 * pi) - (d / 2) * log(det(K) + realmin) - .5 * trace(KinvYYT);