classdef trans_basic
  %TRANS_BASIC Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    szs_in; 
    % [a,b,c]. size of a single input map (for just one instance)
    szs_out; 
    % [aa,bb,cc]. size of a single output map (for just one instance)
  
    is_tr;
    % [1]. Logical. Ture if training, false if testing
    
  end
  
  methods
    function obj = trans_basic()
      obj.is_tr = true;
    end
    
    function data_o = ff(obj,  data_i) 
      error('Should not be called in a base class!\n');
    end
    
    function data_in = deriv_input(obj, data_in, data_out)
      error('Should not be called in a base class!\n');
    end
    
    function obj = deriv_param(obj, data_in, data_out)
    end
    
    function obj = update_param(obj, t)
    % Update parameter
    % Do nothing here
    % Input:
    %   t: [1]. epoch count
      
    end
    
    function obj = init_param(obj, szs_in_)
    % szs_in_: [Hi,Wi,Mi, N].  
      error('Should not be called in a base class!\n');
    end
  end
  
end

