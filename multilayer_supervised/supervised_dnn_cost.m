function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
for l = 1:length(hAct)
   if l == 1 
       hAct{l} = act(bsxfun(@plus, stack{l}.W * data, stack{l}.b), ei.activation_fun);
   elseif l == length(hAct)
       Aout = exp(bsxfun(@plus, stack{l}.W * hAct{l - 1}, stack{l}.b));
       hAct{end} = bsxfun(@rdivide, Aout, sum(Aout));
       clear Aout;
   else
       hAct{l} = act(bsxfun(@plus, stack{l}.W * hAct{l - 1}, stack{l}.b), ei.activation_fun);
   end
end

pred_prob = hAct{end};

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
m = size(data,2);
groundTruth = full(sparse(labels, 1:m, 1));
ceCost = -sum(sum(groundTruth .* log(pred_prob))) / m;

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
Delta = cell(numHidden+1, 1);
for i = length(Delta):-1:1
    if i == length(Delta)
        Delta{i} = -(groundTruth - hAct{i});
    elseif i == 1
        Delta{i} = (stack{i+1}.W' * Delta{i+1}) .* dev(bsxfun(@plus, stack{i}.W * data, stack{i}.b), ei.activation_fun);
    else
%         Delta{i} = (stack{i+1}.W' * Delta{i+1}) .* (hAct{i} .* (1 - hAct{i}));
        Delta{i} = (stack{i+1}.W' * Delta{i+1}) .* dev(bsxfun(@plus, stack{i}.W * hAct{i - 1}, stack{i}.b), ei.activation_fun);
    end
end


%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost = 0;
for i = 1:length(ei.layer_sizes)
    wCost = wCost + (ei.lambda / 2) * norm(stack{i}.W, 'fro')^2;
end

cost = ceCost + wCost;


for i = 1:length(gradStack)
    if i == 1
        gradStack{i}.W = Delta{i} * data' / m + ei.lambda * stack{i}.W;
        gradStack{i}.b = sum(Delta{i}, 2) / m;
    else
        gradStack{i}.W = Delta{i} * hAct{i - 1}' / m + ei.lambda * stack{i}.W;
        gradStack{i}.b = sum(Delta{i}, 2) / m;
    end
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



