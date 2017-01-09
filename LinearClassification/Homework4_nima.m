% Nima Ghorbani
% Linear Classification

%% cleanup/setup d environment;
clear;close all;clc;

load('linearclassification.mat');
xTrain_pos = xTrain(tTrain==1,:); xTrain_neg = xTrain(tTrain==-1,:);

sigmoid = @(x) 1./(1+exp(-x));

%% Linear Discriminant Analysis
% P(t = 1|x) = p(w'x + w0)
% for decision boundary:
% syms x1 x2 w1 w2 w0
% solve([w1 w2]*[x1;x2]+w0==0,x2)

mu_pos = mean(xTrain_pos)'; mu_neg = mean(xTrain_neg)';
cov_pos = cov(xTrain_pos); cov_neg = cov(xTrain_neg);
xTrain_cov = (1/2) * (cov_pos + cov_neg); % equal covariances

w = (xTrain_cov^-1*(mu_pos-mu_neg)); 
w0 = -.5*mu_pos'*xTrain_cov^-1*mu_pos+.5*mu_neg'*xTrain_cov^-1*mu_neg + log(0.5/0.5);

% Training Accuracy
LDA = @(x) 2*(sigmoid(w'*x + w0)>0.5)-1;
LDA_y = @(x) (w'*x + w0); % Distance function

error_count = 0;
for i = 1:length(xTrain)
    if ~(LDA(xTrain(i,:)') == tTrain(i))
        error_count = error_count + 1;
    end           
end
accuracy_LDA_trainig = (1-(error_count/length(xTrain)))*100;

% Test Accuracy
error_count = 0;
for i = 1:length(xTest)
    if ~(LDA(xTest(i,:)') == tTest(i))
        error_count = error_count + 1;
    end           
end
accuracy_LDA_test = (1-(error_count/length(xTest)))*100;

figure(100);hold on;
scatter(xTrain_pos(:,1),xTrain_pos(:,2),'bO');%positive class
scatter(xTrain_neg(:,1),xTrain_neg(:,2),'rO');%negative class
w1=w(1,1);w2=w(2,1);
xboundary = -10:10;
plot(xboundary,-(w0 + w1.*xboundary)./w2,'--k');% plot the decision boundary
title(sprintf('Gaussian Linear Discriminant Analysis - Training Dataset\nTrainig Accuracy: %2.2f%%, and Sample Test Accuracy: %2.2f%%',accuracy_LDA_trainig,accuracy_LDA_test));
xlabel('x');ylabel('y');

%% Quadratic Discriminate Analysis
% QDA Decision Boundary
% syms x1 x2 a1 a2 a3 a4 b1 b2 c
% solve([x1 x2]*[a1 a2;a3 a4]*[x1;x2]+[b1 b2]*[x1;x2]+c==0,x2)

A = -.5*(cov_pos^-1 - cov_neg^-1);
b = (cov_pos^-1*mu_pos - cov_neg^-1*mu_neg);
c = -.5*log(det(cov_pos)/det(cov_neg))-.5*(mu_pos'*cov_pos^-1*mu_pos-mu_neg'*cov_neg^-1*mu_neg);

QDA = @(x) 2*(sigmoid(x'*A*x + b'*x + c)>0.5)-1;
QDA_y = @(x) (x'*A*x + b'*x + c);

error_count = 0;
for i = 1:length(xTrain)
    if ~(QDA(xTrain(i,:)') == tTrain(i))
        error_count = error_count + 1;
    end           
end
accuracy_QDA_training = (1-(error_count/length(xTrain)))*100;

error_count = 0;
for i = 1:length(xTest)
    if ~(QDA(xTest(i,:)') == tTest(i))
        error_count = error_count + 1;
    end           
end
accuracy_QDA_test = (1-(error_count/length(xTest)))*100;

t = -10:10;

a1 = A(1,1);a2 = A(1,2);
a3 = A(2,1);a4 = A(2,2);
b1 = b(1,1);b2=b(2,1);

x21 = [];x22 = [];
for xboundary=t
    x21 = [x21 -(b2 + a2*xboundary + a3*xboundary - (a2^2*xboundary^2 + 2*a2*a3*xboundary^2 + 2*a2*b2*xboundary + a3^2*xboundary^2 + 2*a3*b2*xboundary + b2^2 - 4*a1*a4*xboundary^2 - 4*a4*b1*xboundary - 4*a4*c)^(1/2))/(2*a4)];
    x22 = [x22 -(b2 + a2*xboundary + a3*xboundary + (a2^2*xboundary^2 + 2*a2*a3*xboundary^2 + 2*a2*b2*xboundary + a3^2*xboundary^2 + 2*a3*b2*xboundary + b2^2 - 4*a1*a4*xboundary^2 - 4*a4*b1*xboundary - 4*a4*c)^(1/2))/(2*a4)];
end

figure(225); hold on;
scatter(xTrain_pos(:,1),xTrain_pos(:,2),'bO');%positive class
scatter(xTrain_neg(:,1),xTrain_neg(:,2),'rO');%negative class

plot(t,real(x21),'-r');
plot(t,real(x22),'-b');
title(sprintf('Gaussian Quadratic Discriminant Analysis - Training Dataset\nTrainig Accuracy: %2.2f%%, and Sample Test Accuracy: %2.2f%%',accuracy_QDA_training,accuracy_QDA_test));
xlabel('x');ylabel('y');

%% calculating distance of test set points to decision boundary
% first for QDA
xTest_pos = xTest(tTest==1,:); xTest_neg = xTest(tTest==-1,:);

xTest_pos_QDA_distances = [];
for i = 1:length(xTest_pos)
    xTest_pos_QDA_distances = [ xTest_pos_QDA_distances;QDA_y(xTest_pos(i,:)')];
end

xTest_neg_QDA_distances = [];
for i = 1:length(xTest_neg)
    xTest_neg_QDA_distances = [xTest_neg_QDA_distances;QDA_y(xTest_neg(i,:)')];
end
% Plotting histogram for distances.
figure(204);
[h, b1] = hist(xTest_pos_QDA_distances); hold on;
bar_QDA = bar(b1,h,'histc');set(bar_QDA,'FaceColor','red');set(bar_QDA,'facea',.3);
[h, b2] = hist(xTest_neg_QDA_distances); hold on;
bar_QDA = bar(b2,h,'histc');set(bar_QDA,'FaceColor','blue');set(bar_QDA,'facea',.3);
title('Distribution of distances of the two classes in QDA.')
% Then for LDA

xTest_pos_LDA_distances = [];
for i = 1:length(xTest_pos)
    xTest_pos_LDA_distances = [ xTest_pos_LDA_distances;LDA_y(xTest_pos(i,:)')];
end

xTest_neg_LDA_distances = [];
for i = 1:length(xTest_neg)
    xTest_neg_LDA_distances = [xTest_neg_LDA_distances;LDA_y(xTest_neg(i,:)')];
end
% Plotting histogram for LDA distances.
figure(214);
[h, b1] = hist(xTest_pos_LDA_distances); hold on;
bar_LDA = bar(b1,h,'histc');set(bar_LDA,'FaceColor','red');set(bar_LDA,'facea',.3);
[h, b2] = hist(xTest_neg_LDA_distances); hold on;
bar_LDA = bar(b2,h,'histc');set(bar_LDA,'FaceColor','blue');set(bar_LDA,'facea',.3);
title('Distribution of distances of the two classes in LDA.')

%%% The Shaded area is the probelm which causes the low accuracy of LDA and
%%% QDA. So algorithems like Fisher's Discriminant targets this problem



