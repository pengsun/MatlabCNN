classdef loss_softmax < loss_basic
  %LOSS_SOFTMAX Soft Max, a.k.a. logistic
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function a_loss = calc_loss(obj, a_end, y)
    % ell = \sum_{k=1}^K y_k *(-log(p_k)), 
    % p_k = exp(F_k) / Sigma, Sigma = \sum_{i=1}^K exp(F_i)
      p = logistic_link(a_end);
      a_loss = sum( y .* (-log(p)) ); % sum([K,N])
    end
   
    function d_end = deriv_loss(obj, a_end, y)  
      p = logistic_link(a_end);
      d_end = p - y;
    end
    
  end % methods
  
end

function p = logistic_link(F)
  t = exp(F); % [K, N]
  s = sum(t); % [1, N]
  K = size(F, 1);
  p = t ./ repmat(s, K, 1); % [K, N]
end

