% Nima Ghorbani
% Goal: Oriented PCA
close all; clc; clear all;

% OPCA
load mnist_train
load mnist_test
X0 = train{10}; X1 = train{1};
Test0 = test{10}; Test1 = test{1};

%% 1st Step: PCA to get rid of most obviously unwanted dimensions
X = [X0,X1]; % combined data
UXm=PCA(X);% do PCA and get eigenvectors along the largest variances(eigenvalues)

Y = UXm'*X;
Y0 = Y(:,1:length(X0)); Y1 = Y(:,length(X0)+1:end); % dimensionalit reduced data

CY0 = cov(Y0'); % transpos since each row should be an observation
CY1 = cov(Y1'); % new class condtional covariance

%% 2nd Step: Evaluating noise covariance [mean of two covariances]
CYn = (CY0+CY1)/2;
[UCYn, DCYn, ~] = svd(CYn);
figure(100);imagesc(DCYn);colorbar
title('Eigenvalues of C_{yn} in descending order');

%% 3rd Step) Noise whitening
Z = sqrtm(CYn)^-1*Y;
Z0 = Z(:,1:length(Y0)); Z1 = Z(:,length(Y0)+1:end); % dimensionalit reduced data
CZ0 = cov(Z0'); CZ1 = cov(Z1'); % class conditional covariace of new matriices with isotropic noise
CZn = (CZ0+CZ1)/2; 
[UCZn, DCZn, ~] = svd(CZn); 
figure(101);imagesc(DCZn);colorbar
title('Eigenvalues of C_{zn} showing isotropic noise variance');

%% 4th Step) Another PCA this time over noise whitened data
CZ = cov(Z');
[UZ DZ ~] = svd(CZ);

V2 = UZ(:,1:2); % taking just two dimensions
S = V2'*Z;
S0 = S(:,1:length(Z0)); S1 = S(:,length(Z0)+1:end);

XX = V2'*sqrtm(CYn)*UXm';

X0_kk = XX(1,:); X1_kk = XX(2,:);

figure(102);

subplot223 = subplot(2,2,3);hold on;
plot(subplot223,S0(1,:),S0(2,:),'bO');
plot(subplot223,S1(1,:),S1(2,:),'rO');
legend(subplot223,'S0','S1');

subplot(2,2,1);imagesc(reshape(X0_kk,28,28));%colormap(gray);
subplot(2,2,4);imagesc(reshape(X1_kk,28,28));%colormap(gray);
suptitle('Oriented PCA')
%% Clasification
MS0 = mean(S0,2); MS1 = mean(S1,2);
CS0 = cov(S0'); CS1 = cov(S1');
CS = (1/2) * (CS0 + CS1); % equal covariances

%[opca_Test0,opca_Test1 ]= OPCA(Test0,Test1);
opca_Test0 = XX*Test0; opca_Test1 = XX*Test1;

[S0,S1]= OPCA(X0,X1);

%% Using LDA
% since large dataset MAP can be approximated by MLE

% trying to plot the boundaries from LDA
w = (CS^-1*(MS1-MS0)); 
w0 = -.5*MS1'*CS^-1*MS1+.5*MS0'*CS^-1*MS0 + log(0.5/0.5);

w1=w(1,1);w2=w(2,1);
x1 = -2:0.1:2;
decision_boundary = -(w0 + w1.*x1)./w2;
% figure(302);
% subplot223 = subplot(2,2,3);hold on;
% plot(subplot223,S0(1,:),S0(2,:),'bO');
% plot(subplot223,S1(1,:),S1(2,:),'rO');
% legend(subplot223,'S0','S1');
% subplot(2,2,3);plot(x1,decision_boundary,'--k');% plot the decision boundary


% sigmoid = @(x) 1./(1+exp(-x));
% LDA = @(x) sigmoid(w'*x + w0)>0.5;
% 
% error_count = 0;
% for i = 1:length(S0)
%     if  ~(LDA(S0(:,i))==0)
%         error_count = error_count + 1;
%     end
% end
% for i = 1:length(S1)
%     if  ~(LDA(S1(:,i))==1)
%         error_count = error_count + 1;
%     end
% end
% traning_set_loss = (1-(error_count/length(S)))*100;
% 
% error_count = 0;
% for i = 1:length(opca_Test0)
%     if  ~(LDA(opca_Test0(:,i))==0)
%         error_count = error_count + 1;
%     end
% end
% for i = 1:length(opca_Test1)
%     if  ~(LDA(opca_Test1(:,i))==1)
%         error_count = error_count + 1;
%     end
% end
% test_set_loss = (1-(error_count/length([Test0,Test1])))*100;

%% using MAP
NT0 = length(Test0);NT1 = length(Test1);

Err0 = zeros(1,NT0);
for i=1:10
    Err0(i) = mvnpdf(opca_Test0(:,i),MS1,CS1) - mvnpdf(opca_Test0(:,i),MS0,CS0) >0;
end
Err1 = zeros(1,NT1);
for i=1:NT1
    Err1(i) = mvnpdf(opca_Test1(:,i),MS1,CS1) - mvnpdf(opca_Test1(:,i),MS0,CS0) >=0;
end
test_set_loss = sum(abs(Err0-0)) + sum(abs(Err1-1));

subplot222 = subplot(2,2,2);hold on;
plot(subplot222, opca_Test0(1,:),opca_Test0(2,:),'b*');
plot(subplot222, opca_Test1(1,:),opca_Test1(2,:),'r*');
legend(subplot222, 'T0','T1');
title(subplot222, sprintf('Test Set Loss %2.2f%%',test_set_loss));

NT0 = length(Test0);NT1 = length(Test1);

Err0 = zeros(1,NT0);
for i=1:10
    Err0(i) = mvnpdf(S0(:,i),MS1,CS1) - mvnpdf(S0(:,i),MS0,CS0) >0;
end
Err1 = zeros(1,NT1);
for i=1:NT1
    Err1(i) = mvnpdf(S1(:,i),MS1,CS1) - mvnpdf(S1(:,i),MS0,CS0) >=0;
end
train_set_loss = sum(abs(Err0-0)) + sum(abs(Err1-1));

title(subplot223, sprintf('Training Set Loss %2.2f%%',train_set_loss));














