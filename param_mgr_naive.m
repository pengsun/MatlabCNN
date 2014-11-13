classdef param_mgr_naive < param_mgr_basic
  %PARAM_MGR naive GD 
  %   See the paper "Improvingneural networks by preventing
  %   co-adaptation of feature detectors". Hinton et al.
  
  properties
    alpha;
  end
  
  methods
    
    function obj = param_mgr_naive(varargin)
      if (nargin==0)
        obj.alpha = 0.01;
      elseif (naragin==1)
        obj.alpha = varargin{1};
      else
        error('too many arguments.');
      end
    end % param_mgr_naive
    
    function [obj, theta] = update_param(obj, theta, dtheta, t)
      theta = theta - (obj.alpha)*dtheta;
    end % update_param
  end % methods
  
end % param_mgr_naive