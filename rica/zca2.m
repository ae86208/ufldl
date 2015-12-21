function [Z] = zca2(x)
epsilon = 0;
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)

% x is the input patch data of size
% z is the ZCA transformed data. The dimenison of z = x.

%%% YOUR CODE HERE %%%
x_zeromean = bsxfun(@minus, x, repmat(mean(x), size(x, 1), 1));
sigma = x_zeromean * x_zeromean' / size(x, 2);

[U, S, V] = svd(sigma);

eigen_val = diag(S);

sum_of_eigen_val = sum(eigen_val);
for k = 1: length(eigen_val)
    if sum(eigen_val(1:k)) >= 0.99 * sum_of_eigen_val
        break;
    end
end


Z = U(:, 1:k) * diag(1 ./ sqrt(eigen_val(1:k) + epsilon)) * U(:, 1:k)' * x_zeromean;
end
