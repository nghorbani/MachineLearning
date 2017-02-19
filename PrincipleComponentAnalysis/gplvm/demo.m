close all; close all; clc
[X, labels] = generate_data('helix', 500);
figure, scatter3(X(:,1), X(:,2), X(:,3), 5, labels); title('Original dataset'), drawnow
q = 2;%round(intrinsic_dim(X, 'MLE'));
% disp(['MLE estimate of intrinsic dimensionality: ' num2str(no_dims)]);
% [mappedX, mapping] = compute_mapping(X, 'PCA', no_dims);  
% figure, scatter(mappedX(:,1), mappedX(:,2), 5, labels); title('Result of PCA');
[mappedX]=PCA(X,q);
figure, scatter(mappedX(:,1), mappedX(:,2), 5, labels); title('Result of PCA');

mappedX = gplvm(X, q);
figure, scatter(mappedX(:,1), mappedX(:,2), 5, labels); title('Result of GPLVM');