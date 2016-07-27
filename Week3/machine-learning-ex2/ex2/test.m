

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

%data
%X



%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);
%[m, n]
% Add intercept term to x and X_test
X = [ones(m, 1) X];
%X
% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);
%initial_theta
%y
% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


