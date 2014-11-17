classdef trans_act_sigm < trans_act_basic
  %TRANS_ACT_SIGM Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function [obj, data_o] = ff(obj, data_i, data_o)
      data_o.a = 1 ./ (1 + exp(-data_i.a) );
    end % ff
    
    function data_in = deriv_input(obj, data_in, data_out)
      data_in.d = data_out.a .* (1 - data_out.a) .*...
                  data_out.d; 
    end % deriv_input
    
  end % methods
  
end

