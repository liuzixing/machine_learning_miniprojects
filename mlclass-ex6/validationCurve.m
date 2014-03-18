function [sigma_vec, C_vec, error_val, op_sigma, op_C] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
sigma = [0.01 0.03 0.1 0.3 1 3 10 30]';
C = [0.01 0.03 0.1 0.3 1 3 10 30]';
% You need to return these variables correctly.
% error_train = zeros(length(sigma_vec) * length(C_vec), 1);
error_val = zeros(length(C) * length(sigma), 1);
C_vec = zeros(length(C) * length(sigma), 1);
sigma_vec = zeros(length(C) * length(sigma), 1);
tem = 100000000;
count = 0;
for i = 1:length(sigma)
	for j = 1:length(C)
		count++;
		sigma_vec(count) = sigma(i);
		C_vec(count) = C(j);
		model= svmTrain(X, y, C(j), @(x1, x2) gaussianKernel(x1, x2, sigma(i))); 
		[predictions] = svmPredict(model,Xval);
        [error_val(count)] = mean(double(predictions~= yval));
		if (error_val(count) < tem)
			tem = error_val(count)
			op_sigma = sigma(i)
			op_C = C(j)
			fprintf('%f\t%f\n',op_sigma,op_C);
		end
	end
end









% =========================================================================

end
