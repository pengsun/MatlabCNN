classdef myCNN
  %MYCNN Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    batchsize; % #instances in a batch
    numepochs; % #epochs

    dataArr; 
    % {L+1}. struct storing the data
    %   .a: [Hi,Wi,Mi,N]. Activations. Height,Width,#Maps,#instances
    %   .d: [Hi,Wi,Mi,N]. Deltas. Height,Width,#Maps,#instances
    transArr; 
    % {L}. struct. transArr{i} descirbes the transform from dataArr{i} to
    % dataArr{i+1}; is-a trans_basic
    
    lossType;
    % [1]. class derived from loss_basic

    rL; % [T]. 
    L; % [1]. current loss
  end
  
  methods
    function obj = myCNN(obj)
      obj.batchsize = 50; % #instances in a batch
      obj.numepochs = 1; % #epochs
      
      obj.lossType = loss_le(); % default: square loss
      
      obj.dataArr = {};
      obj.transArr = {};
    end 
    
    function obj = train(obj, X, Y)
    % training  
    % Input:
    %   X: [a,b,N]. training images. N = #instances, [a,b] = image size
    %   Y: [K,N]. targets. K = #classes
    % Output: 
    %  
      % #training instances
      N = size(X, 3);
      
      % #batches
      nb = N / obj.batchsize;
      assert(rem(nb, 1)==0, 'numbatches not an integer');
      bpart = batchPart(N,nb); % instance batches
      
      % initialize transArr
      szs(1) = size(X,1); szs(2) = size(X,2); % size of a single instance/image!
      szs(3) = 1; 
      szs(4) = obj.batchsize;
      obj = init_trans_param(obj, szs);
      
      % initialize dataArr
      obj = init_dataArr(obj, obj.batchsize);
      
      % Stochastic Descent for #epochs
      obj.rL = [];
      for i = 1 : obj.numepochs
        disp(['epoch ' num2str(i) '/' num2str(obj.numepochs)]);
        hps = progressStatus(nb, 10);
        
        tic;
        for j = 1 : nb
          [hps,str] = hps.testChkPnt(j);
          if (~isempty(str)), fprintf('%s...',str); end
          
          ind = bpart.get_ind_from_batch(j);
          batch_x = X(:, :, ind);
          batch_y = Y(:,    ind);
          
          obj = ff(obj, batch_x);
          obj = bp(obj, batch_y);
          obj = update_param(obj, i);
          
          if isempty(obj.rL), obj.rL(1) = obj.L; end
          obj.rL(end + 1) = obj.L;
          %obj.rL(end + 1) = 0.99 * obj.rL(end) + 0.01 * obj.L; % ?
          
        end % for j
        fprintf('\n');
        toc;
        
      end % for i numepochs
    end % train
    
    function [Ypre,dataPreArr] = test(obj,X)
    % Testing
    % Input:
    %  X: [a,b,NN]. testing images
    % Output:
    %  Ypre: [K,NN]. predicted class scores.
    %  dataPreArr: {L+1}. intermediate data
    %
      % Let each trans know the testing context
      for i = 1 : numel(obj.transArr)
        obj.transArr{i}.is_tr = false;
      end
      
      % initialize data array for testing
      N = size(X,3);
      dataPreArr = cell(numel(obj.transArr)+1, 1);
      for i = 1 : numel(dataPreArr)
        dataPreArr{i}.N = N;
      end
      
      % calculate .a for dataTeArr{1} (initialization)
      dataPreArr{1}.a(:,:,1,:) = X;
      % calculate .a for dataTeArr{2:L+1}
      for i = 1 : numel(obj.transArr)
        [obj.transArr{i}, dataPreArr{i+1}] = ff(obj.transArr{i},...
          dataPreArr{i}, dataPreArr{i+1});
      end % for i
      
      % set Ypre
      Ypre = dataPreArr{end}.a;
    end % test
  end % methods
  
  methods % helpers
    function obj = init_trans_param(obj, szs_in)
    % szs_in: [a,b,1]. size for a single instance/image
    % 

      for i = 1 : numel(obj.transArr)
        obj.transArr{i} = init_param(obj.transArr{i}, szs_in);
        
        % update: output size as next input size
        szs_in = obj.transArr{i}.szs_out; 
      end % for i
    end
    
    function obj = init_dataArr(obj, N)
      obj.dataArr = cell(1, numel(obj.transArr)+1);
      for i = 1 : numel(obj.dataArr)
        obj.dataArr{i}.N = N;
      end % i
    end % init_dataArr
    
    function obj = ff(obj, xx)
    % Feed Forward  
    % Input:
    %   xx: [a,b,bs].the batch
    %
    
      % calculate .a for dataArr{1} (initialization)
      obj.dataArr{1}.a(:,:,1,:) = xx;
      
      % calculate .a for dataArr{2:L+1}
      for i = 1 : numel(obj.transArr)
        [obj.transArr{i}, obj.dataArr{i+1}] = ff(obj.transArr{i}, ...
          obj.dataArr{i}, obj.dataArr{i+1} );
      end % for i
    end % ff
    
    function obj = bp(obj, yy)
    % Back Propagation  
      
      obj = calc_deltaLast_and_loss(obj, yy);
    
      % calculate .d for dataArr{L:-1:2}
      for i = numel(obj.transArr) : - 1 : 2
        obj.dataArr{i} = obj.transArr{i}.deriv_input(...
          obj.dataArr{i}, obj.dataArr{i+1});
      end % for i
      
      % the derivative of theta for transArr{L:-1:1}
      for i = numel(obj.transArr) : -1 : 1
        obj.transArr{i} = obj.transArr{i}.deriv_param(...
          obj.dataArr{i}, obj.dataArr{i+1});
      end % for i
    end % update_param
    
    function obj = calc_deltaLast_and_loss(obj, yy)

      % calculate .d for the last data layer dataArr{L+1}, serving as
      % initialization for bp
      obj.dataArr{end}.d = obj.lossType.deriv_loss(...
        obj.dataArr{end}.a, yy);
%       obj.dataArr{end}.d = obj.dataArr{end}.a - yy; % [K, N]
      
      % calculate the loss, by the way
      dataloss = obj.lossType.calc_loss(obj.dataArr{end}.a, yy);
      obj.L = mean(dataloss);
%       N = size(yy,2);
%       tmp = (obj.dataArr{end}.d).^2;
%       obj.L = 0.5 * sum(tmp(:)) / N; 
%       clear tmp;      
    end % calc_deltaLast_and_loss
    
    function obj = update_param(obj, t)
    % update parameters 
      for i = 1 : numel(obj.transArr)
        obj.transArr{i} = update_param(obj.transArr{i}, t);
      end
    end % update_param
  end

end