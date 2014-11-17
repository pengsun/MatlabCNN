classdef vis_cls
  %VIS CNN class-specific visualizer 
  %   Simonyan K. et al. Deep Inside Convolutional Networks: Visualizing
  %   Image Classification Models and Saliency Maps
  
  properties
    T; % [1]. #iterations. for calc_cls_model 
    nu; %[1]. step size. for calc_cls_model
    lambda; % [1]. regularizer. for calc_cls_model
  end
  
  methods
    
    function obj = vis_cls()
      obj.T = 2000;
      obj.nu = 0.0005;
      obj.lambda = 0.005;
    end
    
    function [I,s] = calc_cls_model(obj, hcnn, y, muI)
    % calculate the class model
    % Input:
    %   hcnn: [1]. handle to a trained myCNN model
    %   y: [K,1]. target vector. 1-of-K response
    %   mu: [a,b]. initial value, preferably the mean image
    % Output:
    %   I: [a,b]. the representative image for the target class
    %   s: [K,1]. the class score
      
      %%% initialize the image I
      I = muI;
      
      %%% iterate: gradient ASCENT
      fprintf('Calculating class model: ');
      hps = progressStatus(obj.T, 10);
      for t = 1 : obj.T
        [delI,s] = calc_cls_score_and_grad(hcnn, I, y); % d(Sc(I))/dI
        delObj = delI - 2*obj.lambda*I; % total gradient
        % NOTE: max_{I} Objective, plus the gradient
        I = I + obj.nu * delObj; 
        
        % print
        [hps,str] = hps.testChkPnt(t);
        if (~isempty(str)), fprintf('%s...',str); end
      end
      fprintf('done\n');
    end % clsSalMap
    
    function [M, s] = calc_cls_saliency(obj, hcnn, I, y)
    % calculate the class saliency map
    % Input:
    %   hcnn: [1]. handle to a trained myCNN model
    %   I: [a,b]. the image
    %   y: [K,1]. the target vector. 1-of-K response
    % Output:
    %   M: [a,b]. the signed saliency map
    %   s: [K,1]. the class score      
      [M,s] = calc_cls_score_and_grad(hcnn,I,y);
    end
  end % methods
  
end

function [delI,s] = calc_cls_score_and_grad(hcnn, I, y)

  % behave as if training
  for i = 1 : numel(hcnn.transArr)
    hcnn.transArr{i}.is_tr = true;
  end

  %%% feed foward
  % initialize data array for prediction
  dataPreArr = cell(numel(hcnn.transArr)+1, 1);

  % calculate .a for {1} (initialization)
  dataPreArr{1}.a(:,:,1,:) = I;
  % calculate .a for {2:L+1}
  for i = 1 : numel(hcnn.transArr)
    [hcnn.transArr{i}, dataPreArr{i+1}] = ff(...
      hcnn.transArr{i}, dataPreArr{i} );
  end % for i

  %%% back propagation
  % delta at the last layer: 
  % 1 for the target class score, 0 otherwise, hence precisely the target y
  dataPreArr{end}.d = y;

  % calculate .d for {L:-1:1}
  % IMPORTANT: .d at {1} is also calculated!
  for i = numel(hcnn.transArr) : - 1 : 1
    dataPreArr{i} = hcnn.transArr{i}.deriv_input(...
      dataPreArr{i}, dataPreArr{i+1});
  end % for i

  % awakward workaround
  delI = dataPreArr{1}.d(:,:,1);
  s = dataPreArr{end}.a(:,1);
end