classdef trans_fc < trans_basic
  %TRANS_FC Full Connection
  %   Detailed explanation goes here
  
  properties
    Mo; % #outputs
    
    W; % [Hin,Win,Min, Mo]. 
    b; % [Mo]
    c; % [1]. max-norm for each W(:,:,:,j), j = 1:Mo
    
    dW; % [Hin,Win,Min, Mo]. 
    db; % [Mo]
    
    hpmW; % handle of parameter manager
    hpmb;
  end
  
  methods
    function obj = trans_fc(Mo_)
      obj.Mo = Mo_;
      
      obj.c = inf;
      obj.hpmW = param_mgr_fmwl();
      obj.hpmb = param_mgr_fmwl();
    end
    
    function [obj, data_o] = ff(obj, data_i, data_o) 
    % Feed Forward  
    %
      N = data_i.N;
      Mfe = prod(obj.szs_in(1:end-1)); % #feature
      aa_in = reshape(data_i.a, Mfe, N); % [Mfe,N]
      WW = reshape(obj.W, Mfe, obj.Mo); % [Mfe, Mo]
      
      data_o.a = WW' * aa_in + repmat(obj.b(:),1,N);
      %a_out = sigm( WW' * aa_in + repmat(obj.b(:),1,N) ); % [Mo, N]
    end % ff    
      
    function data_i = deriv_input(obj, data_i, data_o)
      
      % concatenate to vector
      sz = size(obj.W);
      Mout = sz(end);
      MM = prod( sz(1:end-1) );
      WW = reshape(obj.W, MM, Mout); % [MM, Mout]
      
      % calculate [MM, N] = [MM, Mout] * [Mout, N]
      tmp = WW * data_o.d; % 
      
      % restore to multi-dim array
      N = data_o.N;
      szsin = obj.szs_in; szsin(end) = N;
      data_i.d = reshape(tmp, szsin);
    end % deriv_input
    
    function obj = deriv_param(obj, data_i, data_o)
      sz = size(obj.W);
      %Mout = sz(end);
      MM = prod( sz(1:end-1) );
      N = data_o.N;
      
      %%% dW
      % concatenate in vector form
      aa_in = reshape(data_i.a, MM, N); % [MM, N]
      % calculate
      dx = data_o.d;
      tmp = aa_in * dx' ./ N; % [MM,N] * [N, Mout]
      % restore to multi-dim array
      obj.dW = reshape(tmp, sz(:)');
      
      %%% db
      obj.db = mean(dx, 2); % [Mout] = mean([Mout,N], 2)
    end % deriv_param
    
    function obj = update_param(obj, t)
      [obj.hpmW, obj.W] = obj.hpmW.update_param(...
        obj.W, obj.dW, t);
      [obj.hpmb, obj.b] = obj.hpmb.update_param(...
        obj.b, obj.db, t);
      
      if (obj.c < inf)
        obj.W = max_norm(obj.W, obj.c);
      end
      
    end % update_param
    
    function obj = init_param(obj, szs)
    % initialize parameters
      
      % set input map size
      obj.szs_in = szs;
         
      % deduce the output map size
      Mout = obj.Mo;
      obj.szs_out = [Mout, szs(end)];
      
      % randomly initialize the weights: 0-mean gaussian
      f = 0.01;
      obj.W = f*randn([szs(1:end-1),Mout]); 

%       % uniformly random in range [-1,+1]
%       obj.W = 2*(rand([szs(1:end-1),Mout]) - 0.5); 
%       fan_in = prod(szs);
%       fan_out = Mout;
%       obj.W = obj.W * sqrt(6/(fan_in + fan_out));
      
      % ...and set zeros the bias
      obj.b = zeros(Mout,1);
    end % init_param
  end % methods
  
end

function W = max_norm(W, c)
  sz = size(W);
  Mo = sz(end);
  
  % reshape to vector
  W = reshape(W, [prod(sz(1:end-1)), Mo]);
  % clap
  for j = 1 : Mo
    s = norm( W(:,j) );
    if ( s > c )
      W(:,j) = W(:,j) / s * c;
    end
  end % for j
  % reshape to original size
  W = reshape(W, sz);
end