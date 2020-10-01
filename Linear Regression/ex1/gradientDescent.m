function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
i=0;
divJ1=0;
divJ2=0;
for i = 1:m
  temp =((theta'* X(i,:)')-y(i));
  divJ1 = divJ1 +temp;
  divJ2 = divJ2 +temp*X(i,2);
end
divJ1 =(1/(m)) * divJ1;
divJ2 = (1/(m)) * divJ2;
divJ=[divJ1;divJ2];
theta = theta - alpha * divJ;





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
