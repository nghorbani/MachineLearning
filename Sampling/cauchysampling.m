% Nima Ghorbani
% Goal: sampling from a cauchy distribution
clear all; clc; close all;
rng(100);

nsamples = 1000;

invCauchyCDF = @(x) tan(pi*(x-0.5));
CauchyCDF = @(y) (1/pi)*atan(y) + 0.5;
CauchyPDF = @(y) (1/pi)*(1./(1+y.^2));

%% Rejection sampling on the interval -20 and 20
figure(100);
subplot(131);hold on;title('Rejection Sampling');
samples = zeros(1,nsamples);
sIdx = 1;
while sIdx < nsamples
    xCurrent = -20 + 40*rand;
    if rand <= CauchyPDF(xCurrent)
        samples(sIdx) = xCurrent;
        sIdx = sIdx + 1;
    end
end
[bincounts,bincenters] = hist(samples,50);
bincounts = bincounts/diff(bincenters(1:2))/nsamples;
bar(bincenters,bincounts,1,'r');
%histogram(samples,'Normalization','probability')

plot(-20:0.1:20,CauchyPDF(-20:0.1:20),'k--');% theoretical
xlim([-20,20]);
%% Inverse CDF sampling
subplot(132);hold on;title('Inverse CDF Sampling');
samples = rand(1,nsamples);
samples = arrayfun(invCauchyCDF,samples);

[bincounts,bincenters] = hist(samples,nsamples*2);
bincounts = bincounts/diff(bincenters(1:2))/nsamples;
bar(bincenters,bincounts,1,'b');

plot(-20:0.1:20,CauchyPDF(-20:0.1:20),'k--');% theoretical
xlim([-20,20]);
%% Metropolis-Hastings Sampler
subplot(133);hold on;title('MCMC MH Sampler')
proposalDist = @(x,m) (1/sqrt(2*pi))*exp(-0.5*(x-m)^2); % proposal distribution

numThin = 100; % thinning, to get independent samples
samples = zeros(1,nsamples);

for sIdx = 2:nsamples
    xCurrent = normrnd(0,1);%samples(sIdx-1);
    for i = 1:numThin
        xCand = normrnd(xCurrent,1); % normal centered around current x
        alpha = proposalDist(xCurrent,xCand) * CauchyPDF(xCand)/( proposalDist(xCand,xCurrent) * CauchyPDF(xCurrent)); % acceptance ratio
        if alpha >= 1
           xCurrent = xCand;
        elseif rand < alpha
           xCurrent = xCand;
        end
        samples(sIdx) = xCurrent;
    end
end
[bincounts,bincenters] = hist(samples,20);
bincounts = bincounts/diff(bincenters(1:2))/nsamples;
bar(bincenters,bincounts,1,'r');
plot(-20:0.1:20,CauchyPDF(-20:0.1:20),'k--');% theoretical
xlim([-20,20]);
suptitle('Sampling from a cauchy distribution')

