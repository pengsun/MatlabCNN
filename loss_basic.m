classdef loss_basic 
  %TRANS_LOSS Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function a_loss = calc_loss(obj, a_end, y)
    % a_loss = ell(a_end,y) 
    % Input:
    %   a_end: [K, N]. activations at the last dataArr
    %   y: [K, N]. targets
    % Output:
    %   a_loss: [K, N]. the loss
      error('Should not be called in a base class!\n');
    end
    
    function d_end = deriv_loss(obj, a_end, y)
    % d_end = \partial{ell(a_end,y)} / \partial{a_end}  
    % Input:
    %   a_end: [K, N]. activations at the last dataArr
    %   y: [K, N]. targets
    % Output:
    %   d_end: [K, N]. the derivatives    
      error('Should not be called in a base class!\n');
    end
  end
  
end

