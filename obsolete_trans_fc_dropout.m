classdef trans_fc_dropout < trans_fc
  %TRANS_FC Full Connection, with dropout probability 0.5
  %   Detailed explanation goes here
  
  properties
%     Mo; % #outputs
%     
%     W; % [Hin,Win,Min, Mo]. 
%     b; % [Mo]
%     
%     dW; % [Hin,Win,Min, Mo]. 
%     db; % [Mo]
    
    a_in_mask; 
    % [Hi,Win,Min,N]. 0/1 mask, the same size with a_in
  end
  
  methods
    function obj = trans_fc_dropout(Mo_)
      obj = obj@trans_fc(Mo_);
    end
    
    function [data_o,data_i] = ff(obj, data_i)
    % Feed Forward
    %
      if (obj.is_tr) % training
        [data_o,data_i] = ff_tr(obj, data_i);
      else % testing
        [data_o,data_i] = ff_te(obj, data_i);
      end
    end % ff
    
    function [data_o,data_i] = ff_tr(obj, data_i) 
    % Feed Forward for training
    %

      
      % Dropout: make a mask with randomly half of
      % the elements being zero, 
      if (~isfield(data_i,'a_mask'))
        pdropout = 0.5;
        data_i.a_mask = rand( size(data_i.a) );
        data_i.a_mask = double(data_i.a_mask < pdropout);
      end
      
      sz = size(data_i.a);
      N = sz(end);
      Mfe = prod(obj.szs_in); % #feature
      aa_in = (data_i.a_mask .* data_i.a);
      aa_in = reshape(aa_in, Mfe, N); % [Mfe,N]
      WW = reshape(obj.W, Mfe, obj.Mo); % [Mfe, Mo]
      
      % ...and multily it with aa_in
      data_o.a = WW' * aa_in + ...
                 repmat(obj.b(:),1,N);
      %a_out = sigm( WW' * aa_in + repmat(obj.b(:),1,N) ); % [Mo, N]
    

    end % ff_tr 
    
    function [data_o,data_i] = ff_te(obj, data_i)
    % Feed Forward for testing
    %
      sz = size(data_i.a);
      N = sz(end);
      Mfe = prod(obj.szs_in); % #feature
      aa_in = reshape(data_i.a, Mfe, N); % [Mfe,N]
      WW = reshape(obj.W, Mfe, obj.Mo); % [Mfe, Mo]
      
      % Dropout: make an on-average ensemble of thined network
      pdropout = 0.5;
      data_o.a = pdropout.*WW' * aa_in + repmat(obj.b(:),1,N);
      %a_out = sigm( WW' * aa_in + repmat(obj.b(:),1,N) ); % [Mo, N]
    end % ff_te
      
    function data_i = deriv_input(obj, data_i, data_o)
      
      % concatenate to vector
      sz = size(obj.W);
      Mout = sz(end);
      MM = prod( sz(1:end-1) );
      WW = reshape(obj.W, MM, Mout); % [MM, Mout]
      
      % calculate
      %dx = data_o.a.*(1 - data_o.a) .* data_o.d; % [Mout, N]
      dx = data_o.d;
      tmp = WW * dx; % [MM, N] = [MM, Mout] * [Mout, N]
      
      % restore to multi-dim array
      N = size(data_o.d, 2);
      data_i.d = reshape(tmp, [obj.szs_in,N]);
      data_i.d = data_i.d .* data_i.a_mask; % dropout TODO: check it
    end % deriv_input
    
    function obj = deriv_param(obj, data_i, data_o)
      sz = size(obj.W);
      %Mout = sz(end);
      MM = prod( sz(1:end-1) );
      N = size(data_o.d, 2);
      
      %%% dW
      % concatenate in vector form
      aa_in = data_i.a .* data_i.a_mask; % dropout TODO: check it
      aa_in = reshape(aa_in, MM, N); % [MM, N]
      % calculate
      %dx = data_o.a .* (1 - data_o.a) .* data_o.d; % [Mout, N]
      dx = data_o.d;
      tmp = aa_in * dx' ./ N; % [MM,N] * [N, Mout]
      % restore to multi-dim array
      obj.dW = reshape(tmp, sz(:)');
      %net.dffW = net.od * (net.fv)' / size(net.od, 2);
      
      %%% db
      obj.db = mean(dx, 2); % [Mout] = mean([Mout,N], 2)
      %net.dffb = mean(net.od, 2);
    end % deriv_param
    
  end % methods
  
end

