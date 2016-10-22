function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
% =============================================================

% grad = grad(:);
hX = sigmoid(X*theta);					% hypothesis answer
lX = log(hX);						% log of hX
s1 = (y')*lX;						% first term in sum for cost function
lonemX = log(ones(size(hX)) - hX);			% log of one minus hX
onemy = ones(size(y)) - y;				% one minus y
s2 = onemy' * lonemX;					% second term in sum for cost function	
J_unregularized = (-1/m) * (s1 + s2);			% cost function without regularization
theta_ = theta(2:end);					% theta_ is theta without theta(1)
J_regularization_term = (lambda/(2*m))*sum(theta_.^2);	% term to be added to unregularized cost function
J = J_unregularized + J_regularization_term;		% obtaining regularized cost function
grad_unregularized = (1/m) * X' * (hX - y);				
grad_regularization_term = [0; theta_ .* (lambda/m)];
grad = grad_unregularized + grad_regularization_term; 


end



