function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



X = [ones(m,1) X]; % X = a1 (input layer activation)
z2 = X * Theta1';
a2 = sigmoid(z2);

% add a2 bias units
a2 = [ones(m,1) a2];
z3 = a2 * Theta2';

%[max_vals, H] = max(z3, [], 2); 


fprintf("\nX size after adding bias -------------- \n")
size(X)
fprintf("\Theta1 size -------------- \n")
size(Theta1)
fprintf("\Theta2 size -------------- \n")
size(Theta2)

fprintf("\nz2 size -------------- \n")
size(z2)
fprintf("\na2 size after adding bias -------------- \n")
size(a2)

fprintf("\nz3 size -------------- \n")
size(z3)
a3 = H = sigmoid(z3) >= 0.5;


fprintf("\na3  -------------- \n")
size(a3)
fprintf("\nH  -------------- \n")
size(H)
fprintf("\ny  -------------- \n")
size(y)

y_Label_Values = y;
y = zeros(m, num_labels);

y(sub2ind(size(y), 1:size(y_Label_Values, 1), y_Label_Values')) = 1;

fprintf("\n y_Label_Values(2000)");
y_Label_Values(2000)
fprintf("\ny(2000)");
y(2000, :)



size(y)

 
%J = sum(-y' * log(H) - (1 - y) * log(1 - H))

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
