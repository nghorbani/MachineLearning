function [S0,S1]=OPCA(data0,data1,preserved_variance)
%%Oriented PCA
if nargin<3, preserved_variance=0.9; end
data = [data0,data1];
% First ordinary PCA to get rid of most obviously unwanted dimensions
UXm=PCA(data,preserved_variance);% do PCA and get eigenvectors along the largest variances(eigenvalues)
Y = UXm'*data;
Y0 = Y(:,1:length(data0)); Y1 = Y(:,length(data0)+1:end); % dimensionalit reduced data

CY0 = cov(Y0'); % transpos since each row should be an observation
CY1 = cov(Y1'); % new class condtional covariance

% Second Evaluating noise covariance
CYn = (CY0+CY1)/2;
[~, DCYn, ~] = svd(CYn);

% Third noise whitening
Z = sqrtm(CYn)^-1*Y;
Z0 = Z(:,1:length(Y0)); Z1 = Z(:,length(Y0)+1:end); % dimensionalit reduced data
CZ0 = cov(Z0'); CZ1 = cov(Z1'); % class conditional covariace of new matriices with isotropic noise
CZn = (CZ0+CZ1)/2; 
[~, DCZn, ~] = svd(CZn); 
% c) 3rd Step) doing another PCA this time over noise whitened data
CZ = cov(Z');
[UZ DZ ~] = svd(CZ);
%
V2 = UZ(:,1:2); % because output of svd is sorted for square of eigen values
S = V2'*Z;
S0 = S(:,1:length(Z0)); S1 = S(:,length(Z0)+1:end);
   
end