function prob = softmax_prob(input)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% prob = zeros(size(theta, 2), size(X, 2));

prob = exp(input);
sum_of_prob = sum(prob);

for i= 1:size(sum_of_prob, 2)
    prob(:, i) = prob(:, i) / (sum_of_prob(i) + 1);

end
end
