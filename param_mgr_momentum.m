classdef param_mgr_momentum < param_mgr_basic
  %PARAM_MGR SGD with momentum
  %   See the paper "Improvingneural networks by preventing
  %   co-adaptation of feature detectors". Hinton et al.
  
  properties
    epsilon0; % for learning rate 
    f; % for learning rate
    
    pi; % for momentum
    pf; % for momentum 
    T; % for momentum
    
    del_theta; % incremental at last iteration 
  end
  
  methods
    function obj = param_mgr_momentum()
      obj.epsilon0 = 10.0;
      obj.f = 0.998;
      
      obj.pi = 0.5;
      obj.pf = 0.99;
      obj.T = 500;
      
      obj.del_theta = 0.0;
    end
    
    function [obj, theta] = update_param(obj, theta, dtheta, t)
      % update using gradient with momentum
      pt = get_pt(obj, t);
      epsilont = (obj.epsilon0) .* ((obj.f).^t);
      del = pt * obj.del_theta - (1-pt)*epsilont* dtheta;
      theta = theta + del;
      
      % record
      obj.del_theta = del;
    end % update_param
    
    function pt = get_pt(obj, t)
      if (t < obj.T)
        tmp = t/obj.T;
        pt = tmp* obj.pi + (1-tmp)* obj.pf;
      else
        pt = obj.pf;
      end
    end
  end
  
end



