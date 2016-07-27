
clear ; close all; clc

data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

%plotData(X, y);

% Put some labels 
%hold on;

% Labels and Legend
%xlabel('Microchip Test 1')
%ylabel('Microchip Test 2')

% Specified in plot order
%legend('y = 1', 'y = 0')
%hold off;


fprintf("\nX size ---------------\n")
size(X)
X = mapFeature(X(:,1), X(:,2));