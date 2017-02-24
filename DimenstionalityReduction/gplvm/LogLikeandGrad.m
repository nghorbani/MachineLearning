function [L, dLdP] = LogLikeandGrad(p, Y,X)
    dLdP = LogLikeGrad(p, Y,X);
    dLdP = dLdP';
    L = LogLike(p, Y,X);
end