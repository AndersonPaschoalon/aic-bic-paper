function [theta, J_history] = gradientDescent(X, y, theta, learning_rate, num_iters)
% gradientDescent - Performs gradient descent to learn theta
% theta = GRADIENTDESENT(X, y, theta, learning_rate, num_iters) updates theta by 
% taking num_iters gradient steps with learning rate learning_rate
% 
% Syntax: [theta, J_history] = gradientDescent(X, y, theta, learning_rate, num_iters)
%
% Inputs:
%   X - data 
%   y - data 
%   theta - init learn vector
%   learning_rate - learning rate 
%   num_iters - number of iterations
%
% Outputs:
%    theta - learn vector
%    J_history - cost history 
%
% Example: 
%    [theta, J_history] = gradientDescent(X, y, theta, learning_rate, num_iters)
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% Author: Anderson Paschoalon
% email: anderson.paschoalon@gmail.com
% Sep 2018: Last revision: 16-Sep-2018 
% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    %vars
    temp1 = 0;
    temp2 = 0;
    delta1 = 0;
    delta2 = 0;
    x = X(:,2);

    %slope
    delta1 = sum(theta(1) + theta(2)*x - y)/m;
    delta2 = sum( (theta(1) + theta(2)*x - y).*x )/m;

    %linear regression
    temp1 = theta(1) - learning_rate*delta1;
    temp2 = theta(2) - learning_rate*delta2;
    
    theta = [temp1;temp2];

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    fprintf('.');        
end

theta = real(theta);

fprintf('\n');
end
