function [X, Y] = make_rand_inst(H,W,K, N)
%MAKE_RAND_INST Summary of this function goes here
%   Detailed explanation goes here

  X = rand(H,W,N);
  Y = zeros(K,N);
  for n = 1 : N
    ii = randi(10,1);
    Y(ii,n) = 1;
  end

end

