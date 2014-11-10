classdef trans_respnorm < trans_basic
  %TRANS_RESPNORM Local Response Normalization
  %   See Krizhevsky A et al, ImageNet Classification with Deep 
  %   Convolutional  Neural Networks, NIPS2012
  
  properties
    wnsz; % [1]. window size
    
    k;
    alpha;
    beta;
  end
  
  methods
    function obj = trans_respnorm(varargin)
      obj.wnsz = 3;
      obj.k = 2;
      obj.alpha = 1e-4;
      obj.beta = 0.75;
      
      if (nargin == 1)
        obj.wnsz = varargin{1};
      elseif (nargin == 4)
        obj.wnsz = varargin{1};
        obj.k = varargin{2};
        obj.alpha = varargin{3};
        obj.beta = varargin{4};
      end

    end % trans_connorm
    
    function data_o = ff(obj, data_i) 
%       data_o.a = data_i.a;
      
      %
%       dim_prm = [3,1,2,4];
%       aip = permute(data_i.a,  dim_prm); % [H,W,M,N] -> [M, H,W,N]
%       aop = convn(aip, ones(obj.wnsz,1),  'same');
%       data_o.a = ipermute(aop, dim_prm); % [M,H,W,N] -> [H,W,M,N]       
      
      % a_i -> b1: square
      data_o.b1 = (data_i.a).*(data_i.a);
      
      % b1 -> b2: average as convolution
      dim_prm = [3,1,2,4];
      b1p = permute(data_o.b1,  dim_prm); % [H,W,M,N] -> [M, H,W,N]
      b2p = convn(b1p, ones(obj.wnsz,1),  'same');
      data_o.b2 = ipermute(b2p, dim_prm); % [M,H,W,N] -> [H,W,M,N] 
      
      % b2 -> b3
      data_o.b3 = (obj.k + obj.alpha .* data_o.b2).^(obj.beta);
      
      % b3 -> a_o
      data_o.a = data_i.a ./ data_o.b3;
    end % ff
    
    function data_i = deriv_input(obj, data_i, data_o)
%       data_i.d = data_o.d;
      
%         %
%         dim_prm = [3,1,2,4]; % [H,W,M,N] -> [M, H,W,N]
%         dop = permute(data_o.d, dim_prm);
%         dip = convn(dop, ones(obj.wnsz, 1), 'same');
%         data_i.d = ipermute(dip, dim_prm);

      % d(b3)/d(b2) 
      b3_b2 = obj.alpha * obj.beta * ...
              (obj.k + obj.alpha .* data_o.b2).^(obj.beta-1);
      % d(b2)/d(b1)
      dim_prm = [3,1,2,4]; % [H,W,M,N] -> [M, H,W,N]
      b2p = permute(data_o.b2, dim_prm);
      b2_b1p = convn(b2p, ones(obj.wnsz, 1), 'same');
      b2_b1 = ipermute(b2_b1p, dim_prm);      
      % d(b1)/d(ai)
      b1_ai = (data_i.a).^2;
      
      % d(b3)/d(ai) = d(b1)/d(ai) * d(b2)/d(b1) * d(b3)/d(b2) 
      b3_ai = b1_ai .* b2_b1 .* b3_b2;
      
      % di = ( 1/b3 - ai/(b3^2) * d(b3)/d(ai) ) .* do;
      data_i.d = (1./data_o.b3 - data_i.a ./ ((data_o.b3).^2) .* b3_ai) .*...
                 data_o.d;
    end % deriv_input
      
    function obj = init_param(obj, szs)
      % make sure it's 2D CNN
      assert(numel(szs) == 4); % [Height, width, #maps, #instances]
      % set input map size
      obj.szs_in = szs;
      % deduce the output map size
      obj.szs_out = szs;
    end % init_param
    
  end % methods
  
end