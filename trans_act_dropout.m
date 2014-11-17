classdef trans_act_dropout < trans_act_basic
  %TRANS_ACT_DROPOUT Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    p; 
    % [1]. for dropout. 
    % probability/portion of the preserved non-zero elements 
  end
  
  properties
    a_mask;
  end
  
  methods
    function obj = trans_act_dropout(varargin)
    % trans_act_dropout()
    % trans_act_dropout(p)
    % Input:
    %   p: [1]. portion of the preserved non-zero elements
      if (nargin==0)
        obj.p = 0.5;
      else
        obj.p = varargin{1};
      end
    end
    
    function [obj, data_o] = ff(obj, data_i, data_o)
      % identity transform
      data_o.a = data_i.a;
      % dropout
      if (obj.is_tr) % training
        data_o.a = data_i.a;
        data_o.a(obj.a_mask) = 0;
      else % testing
        data_o.a = obj.p .* data_i.a; 
      end
    end % ff
    
    function data_i = deriv_input(obj, data_i, data_o)
      
      data_i.d = data_o.d;
      data_i.d(obj.a_mask) = 0;
    end % deriv_input
    
    function obj = update_param(obj, t)
    % update parameter:
    % simply recreate the dropout mask

      obj.a_mask = (rand(obj.szs_in) > obj.p);
    end % update_param_grad
    
    function obj = init_param(obj, szs_in_)
    % initialize parameters
      
      obj = init_param@trans_act_basic(obj, szs_in_);
      
      % dropout: the mask
      obj.a_mask = (rand(szs_in_) > obj.p);
    end % init_param
    
  end % methods
  
end

