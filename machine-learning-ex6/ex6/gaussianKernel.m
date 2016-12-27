function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%


diff_vector = x1 .- x2;
diff_vector_sq = diff_vector .^ 2;
required_num = sum(diff_vector_sq);
required_denom = 2*sigma*sigma;
required_power = -required_num/required_denom;
sim = e^required_power;



% =============================================================
    
end
