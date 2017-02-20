close all; close all; clc
[Y, labels] = generate_data('helix', 1000);
[n,d] = size(Y);
figure, scatter3(Y(:,1), Y(:,2), Y(:,3), 5, labels); title('Original dataset'), drawnow
q = 2;%round(intrinsic_dim(X, 'MLE'));
% disp(['MLE estimate of intrinsic dimensionality: ' num2str(no_dims)]);
% [mappedX, mapping] = compute_mapping(X, 'PCA', no_dims);  
% figure, scatter(mappedX(:,1), mappedX(:,2), 5, labels); title('Result of PCA');
[Xpca]=PCA(Y,q);
figure, scatter(Xpca(:,1), Xpca(:,2), 5, labels); title('Result of PCA');

Xgplvm = gplvm(Y, q);
figure, scatter(Xgplvm(:,1), Xgplvm(:,2), 5, labels); title('Result of GPLVM');


% Yreverse0 = mvnrnd(zeros(n,d),cov(Y));
% 
% [Ygplvmreverse,lgplvmreverse,sigmafgplvmreverse] = gplvm_reversemap(Xgplvm,Yreverse0,d);
% figure, scatter3(Ygplvmreverse(:,1), Ygplvmreverse(:,2), Ygplvmreverse(:,3), 5, labels); title('Original dataset'), drawnow
