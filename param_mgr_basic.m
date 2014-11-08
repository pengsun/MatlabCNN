classdef param_mgr_basic
  %PARAM_MGR Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    t; % epoch count
  end
  
  methods
    
    function [obj, theta] = update_param(obj, theta, dtheta)
      error('Should not be called in a base class!\n');
    end % update_param    
    
  end
  
end

