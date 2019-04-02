function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
y_mat=zeros(num_labels,m);
for i=1:m,
  y_mat(y(i),i)=1;
end;

 
tmp_X=ones(size(X,1),size(X,2)+1);
tmp_X(1:end,2:end)=X;
tmp_a1=ones(size(X,1),size(Theta1,1)+1);
a1=tmp_X*Theta1';
z1=ones(size(X,1),size(Theta1,1)+1);
z1(1:end,2:end)=sigmoid(a1);
tmp_a1(1:end,2:end)=sigmoid(a1);
HX=tmp_a1*Theta2';
J_sum=0;
tmp_ans=0;
for i=1:num_labels,
  tmp_y=y_mat(i:i,1:end);
  tmp_h=sigmoid(HX)'(i:i,1:end);
  tmp=(-tmp_y*log(tmp_h')-(1-tmp_y)*log(1-tmp_h'));
  tmp_ans=tmp_ans+tmp;
end;

%_sum=sum(sum((tmp_ans)));
J=tmp_ans/m+(sum(sum(Theta1(1:end,2:end).*Theta1(1:end,2:end)))+sum(sum(Theta2(1:end,2:end).*Theta2(1:end,2:end))))*lambda/(2*m);
%J_sum=J_sum+();

%delta2=sigmoid(HX)'-y_mat;

%delta1=Theta2(1:end,2:end)'*delta2.*sigmoidGradient(tmp_a1(1:end,2:end)');
%Theta1_grad=delta1*tmp_X/m+(lambda/m)*Theta1;

%Theta2_grad=delta2*tmp_a1/m+(lambda/m)*Theta2;
z2=(tmp_X*Theta1');
a2=ones(size(z2,1),size(z2,2)+1);
a2(1:end,2:end)=sigmoid(z2);
z3=HX;
a3=sigmoid(z3);

delta3=a3-y_mat';

d1=0;
d2=0;

for t=1:m,
  tmp_z2=z2(t:t,1:end);
  tmp_delta3=delta3(t:t,1:end);
  tmp_z2=[1,tmp_z2];
  
  tmp_delta2=Theta2'*tmp_delta3'.*sigmoidGradient(tmp_z2');
  tmp_a3=[1,a3(t:t,1:end)];
  d2=d2+tmp_delta3'*a2(t:t,1:end);
  d1=d1+tmp_delta2(2:end)*tmp_X(t:t,1:end);

end;
tmp_Theta1=Theta1;
tmp_Theta1(1:end,1:1)=0;
tmp_Theta2=Theta2;
tmp_Theta2(1:end,1:1)=0;
Theta1_grad=d1/m+(lambda/m)*tmp_Theta1;

Theta2_grad=d2/m+(lambda/m)*tmp_Theta2;









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
