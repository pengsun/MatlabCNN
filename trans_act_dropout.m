classdef trans_act_dropout < trans_act_basic
  %TRANS_ACT_DROPOUT Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    a_mask;
  end
  
  methods
    function data_o = ff(obj, data_i)
      % identity transform
      data_o.a = data_i.a;
      % dropout
      p = 0.5;
      if (obj.is_tr) % training
        data_o.a = data_i.a;
        data_o.a(obj.a_mask) = 0;
      else % testing
        data_o.a = p .* data_i.a; 
      end
    end % ff
    
    function data_i = deriv_input(obj, data_i, data_o)
      
      data_i.d = data_o.d;
      data_i.d(obj.a_mask) = 0;
    end % deriv_input
    
    function obj = update_param_grad(obj, alpha)
      % simply recreate the dropout mask
      p = 0.5;
      obj.a_mask = (rand(obj.szs_in) > p);
    end % update_param_grad
    
    function obj = init_param(obj, szs)
    % initialize parameters
      
      obj = init_param@trans_act_basic(obj, szs);
      
      % dropout: the mask
      p = 0.5;
      obj.a_mask = (rand(szs)>p);
    end % init_param
    
  end % methods
  
end

