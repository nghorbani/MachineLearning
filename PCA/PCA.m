function [Um]=PCA(data,preserved_variance)
if nargin<2;preserved_variance=0.9;end
M = mean(data,2);
data_centered = data-M*ones(1,length(data)); % reducing mean from every element
N = size(data_centered,1);
Cov_data = (1/N)*(data_centered*data_centered');

[U, D, V] = svd(Cov_data);
%[U, D] = eig(Cov_X);
diag_D = diag(D); %
sum_D = sum(diag_D); % total variance
variance_sum_upto_m = zeros(1,N);
for m=1:N
    variance_sum_upto_m(m) = sum(diag_D(1:m));
end
variance_sum_upto_m = variance_sum_upto_m/sum_D;

PCs = find(variance_sum_upto_m>preserved_variance,1);
Um = U(:,1:PCs);
        
% figure(104);hold on;
% plot(1:N,variance_sum_upto_m,'k');
% plot(PCs,variance_sum_upto_m(PCs),'Or');
% title('Dimensions preserveing upto 90% of variance');
% xlabel('Dimensions');ylabel('Variance perserved');

end
