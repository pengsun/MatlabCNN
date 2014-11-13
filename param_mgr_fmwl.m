classdef param_mgr_fmwl < param_mgr_basic
  %PARAM_MGR_fmwl SGD with fixed momentum, weight decay and learning rate
  %   See the paper "ImageNet classification with Deep COnvolutional Neural
  %   Network". Krizhevsky et al, where a variable learning 
  %   rate is used. The same for other settings.
  
  properties
    epsilon; % for learning rate 
    wd; % for weight decay
    p; % for momentum
    
    del_theta; % incremental at last iteration 
  end
  
  methods
    function obj = param_mgr_fmwl()
      obj.epsilon = 0.01;
      obj.wd = 0.0005;
      obj.p = 0.9;
      
      obj.del_theta = 0.0;
    end
    
    function [obj, theta] = update_param(obj, theta, dtheta, t)
      % update using gradient with momentum

      del = obj.p * obj.del_theta ... % momentum of increment at last iteration
            - obj.wd * obj.epsilon * theta ... % weight decay
            - obj.epsilon * dtheta; % gradient
      theta = theta + del;
      
      % record
      obj.del_theta = del;
    end % update_param
    
  end % methods
  
end