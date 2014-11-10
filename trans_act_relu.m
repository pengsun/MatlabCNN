classdef trans_act_relu < trans_act_basic
  %TRANS_ACT_RELU Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function data_o = ff(obj, data_i)
      data_o.a = max(0, data_i.a);
    end % ff
    
    function data_in = deriv_input(obj, data_in, data_out)
    % y = max(x,0); dy/dx = 1 if (x>0), 0 otherwise  

      data_in.d = data_out.d;
      data_in.d( data_in.a < 0 ) = 0;
    end % deriv_input    
  end
  
end