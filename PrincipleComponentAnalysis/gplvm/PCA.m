function [mapped]=PCA(data,q)
Cov_data = cov(data);

[U, ~, ~] = svd(Cov_data);

Um = U(:,1:q);
mapped = data*Um;

end
