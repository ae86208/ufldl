function [ dx ] = dev(x, type)
%DEV derivative of designated functions
%   sigmoid, tanh, and rectified linear included
switch type
    case 'logistic'
        dx = bsxfun(@times, sigmoid(x), (1 - sigmoid(x)));
    case 'tanh'
%         dx = sech(x).^2;
        dx = bsxfun(@power, sech(x), 2);
    case 'reLU'
        dx = bsxfun(@max, zeros(size(x)), x>0);
end

end

