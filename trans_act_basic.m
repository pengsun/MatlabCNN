classdef trans_act_basic < trans_basic
  %TRANS_ACT_BASIC Activation Functions. Point-wise calculation
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function [obj, data_o] = ff(obj, data_i) 
      error('Should not be called in a base class!\n');
    end
    
    function data_in = deriv_input(obj, data_in, data_out)
      error('Should not be called in a base class!\n');
    end
    
    function obj = deriv_param(obj, data_in, data_out)
    % do nothing  
    end
    
    function obj = update_param_grad(obj, alpha)
    % do nothing  
    end
    
    function obj = init_param(obj, szs_in_)
    % szs: [a,b,c].  
    
      % set input map size
      obj.szs_in = szs_in_;

      % deduce the output map size, always the same with input
      obj.szs_out = szs_in_;
    end        
  end
  
end

