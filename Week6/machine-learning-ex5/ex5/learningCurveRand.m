function [error_train, error_val] = learningCurveRand(X, y, Xval, yval, lambda, i)


m = size(X, 1);
numberOfTests = 50;
error_train = zeros(numberOfTests, 1);
error_val = zeros(numberOfTests, 1);

for j = 1:numberOfTests

	rand_indices = randperm(m);

	X_rand = X(rand_indices(1:i), :);			
	y_rand = y(rand_indices(1:i), 1);

	rand_indices = randperm(m);
	Xval_rand = Xval(rand_indices(1:i), :);
	yval_rand = yval(rand_indices(1:i), 1);

	lambda_null = 0;
	theta = trainLinearReg(X_rand, y_rand, lambda);
	error_train(j) = linearRegCostFunction(X_rand, y_rand, theta, lambda_null);
	error_val(j) = linearRegCostFunction(Xval_rand, yval_rand, theta, lambda_null);
end

error_train = mean(error_train);
error_val = mean(error_val);

plot(1:m, error_train, 1:m, error_val);
xlabel('Number of training examples');
ylabel('Error');
axis([0 13 0 100]);
legend('Train', 'Cross Validation');

end