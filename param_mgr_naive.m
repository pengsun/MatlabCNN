classdef param_mgr_naive < param_mgr_basic
  %PARAM_MGR naive GD 
  %   See the paper "Improvingneural networks by preventing
  %   co-adaptation of feature detectors". Hinton et al.
  
  properties
  end
  
  methods
    
    function [obj, theta] = update_param(obj, theta, dtheta, t)
      theta = theta - dtheta;
    end % update_param
  end
  
end

