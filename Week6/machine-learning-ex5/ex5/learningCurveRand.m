function [error_train, error_val] = learningCurveRand(X, y, Xval, yval, lambda)


m = size(X, 1);
numberOfTests = 20;
lambda_null = 0;
error_train = zeros(m, 1);
error_val = zeros(m, 1);

%error_train = zeros(m, 1);
%error_val = zeros(m, 1);

%i = number of training examples
for n = 1:numberOfTests
	for i = 1:m
		rand_indices = randperm(m);

		X_rand = X(rand_indices(1:i), :);			
		y_rand = y(rand_indices(1:i), 1);
		
		rand_indices = randperm(size(Xval, 1));
		Xval_rand = Xval(rand_indices(1:i), :);
		yval_rand = yval(rand_indices(1:i), 1);
		
		theta = trainLinearReg(X_rand, y_rand, lambda);
		error_train(i) += linearRegCostFunction(X_rand, y_rand, theta, lambda_null);
		error_val(i) += linearRegCostFunction(Xval_rand, yval_rand, theta, lambda_null);
	end
end

error_train = error_train ./ numberOfTests;
error_val = error_val ./ numberOfTests;

plot(1:m, error_train, 1:m, error_val);
xlabel('Number of training examples');
ylabel('Error');
axis([0 13 0 100]);
legend('Train', 'Cross Validation');

end