classdef loss_le < loss_basic
  %TRANS_LOSS Least Square
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function a_loss = calc_loss(obj, a_end, y)
      tmp = a_end - y;
      a_loss = 0.5 .* sum( tmp.^2 );
    end
    
    function d_end = deriv_loss(obj, a_end, y)  
      d_end = a_end - y;
    end
    
  end
  
end

