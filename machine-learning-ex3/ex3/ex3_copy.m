
clear ; close all; clc;

input_layer_size = 400
num_labels = 10


%% =========== Part 1: Loading and Visualizing Data =============

fprintf('Loading and visualizing data.... \n');
load('ex3data1.mat');

m = size(X,1);
%% m = 5000
n = size(X,2);
%% n = 400

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;



