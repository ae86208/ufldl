function [ac] = act(x, type)
%ACT activate function for neural nets
%    sigmoid, tanh, and rectified linear included
switch type
    case 'logistic'
        ac = sigmoid(x);
    case 'tanh'
        ac = tanh(x);
    case 'reLU'
        ac = bsxfun(@max, zeros(size(x)), x);
end


end

