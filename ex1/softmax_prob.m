function prob = softmax_prob(theta, X)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% prob = zeros(size(theta, 2), size(X, 2));

prob = exp(theta' * X);
sum_of_prob = sum(prob);

for i= 1:size(sum_of_prob, 2)
    prob(:, i) = prob(:, i) / (sum_of_prob(i) + 1);

end
end
