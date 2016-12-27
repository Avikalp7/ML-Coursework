%% Initialization
clear ; close all; clc

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex5data1.mat');

% m = Number of examples
m = size(X, 1);
size(Xtest)
lambda = 3;
theta = trainLinearReg([ones(m,1) X], y, lambda);
testError = linearRegCostFunction([ones(size(Xtest, 1), 1) Xtest], ytest, theta, 0);
fprintf('Test error is : ');
testerror

fprintf('Program paused. Press enter to continue.\n');
pause;

clear ; close all; clc
