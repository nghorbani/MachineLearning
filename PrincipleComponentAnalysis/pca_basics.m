%% Baiscs of the Principle Component Analysis
% MNIST & PCA
load mnist_train
X0 = train{10}; X1 = train{1};

% a) class conditional means and variances
M0 = mean(X0,2); M1 = mean(X1,2); % emperical means
X0_centered = X0-M0*ones(1,length(X0)); % centering by mean (variance)
X1_centered = X1-M1*ones(1,length(X1)); 

N0 = size(X0_centered,1);
N1 = size(X1_centered,1);

Cov_X0 = (1/N0)*(X0_centered*X0_centered'); Cov_X1 = (1/N1)*(X1_centered*X1_centered'); %covariances
figure(103);
subplot(1,2,1);imagesc(Cov_X0);colorbar
title('Covariance of X_0');
xlabel('Dimensions');ylabel('Trials');

subplot(1,2,2);imagesc(Cov_X1);colorbar
title('Covariance of X_1');
xlabel('Dimensions');
% b) PCA
X = [X0,X1];
M = mean(X,2);
X_centered = X-M*ones(1,length(X)); % reducing mean from every element
N = size(X_centered,1);
Cov_X = (1/N)*(X_centered*X_centered');

[U, D, V] = svd(Cov_X); 

diag_D = diag(D);
sum_D = sum(diag_D); % total variance
variance_sum_upto_m = zeros(1,N);
for m=1:N
    variance_sum_upto_m(m) = sum(diag_D(1:m));
end
variance_sum_upto_m = variance_sum_upto_m/sum_D;

PC = find(variance_sum_upto_m>0.9,1,'first');
Um = U(:,1:PC);
        
figure(104);hold on;
plot(1:N,variance_sum_upto_m,'k');
plot(PC,variance_sum_upto_m(PC),'Or');
title('Dimensions preserveing upto 90% of variance');
xlabel('Dimensions');ylabel('Variance perserved');

Y = Um'*X;
Y0 = Y(:,1:length(X0));
Y1 = Y(:,length(X0)+1:end);

M0 = mean(Y0,2); M1 = mean(Y1,2); % emperical means
Y0_centered = Y0-M0*ones(1,length(Y0)); % centering by mean (variance)
Y1_centered = Y1-M1*ones(1,length(Y1)); 

N0 = size(Y0_centered,1);
N1 = size(Y1_centered,1);

Cov_Y0 = (1/N0)*(Y0_centered*Y0_centered'); Cov_Y1 = (1/N1)*(Y1_centered*Y1_centered'); %covariances
figure(105);
subplot(1,2,1);imagesc(Cov_Y0);colorbar
title('Covariance of Y_0');
xlabel('Dimensions');ylabel('Trials');

subplot(1,2,2);imagesc(Cov_Y1);colorbar
title('Covariance of Y_1');
xlabel('Dimensions');

%X_hat = Um*Y;
%X0_hat = X_hat(:,1:length(X0));
%X1_hat = X_hat(:,length(X0)+1:end);
%reconstruction_error_X0 = mean(sum( (X0-X0_hat).^2,1 )./ sum( X0.^2))
%reconstruction_error_X1 = mean(sum( (X1-X1_hat).^2,1 )./ sum( X1.^2))