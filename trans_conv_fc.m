classdef trans_conv_fc < trans_basic
  %TRANS_CONV_FC Convolution, fully connected feature maps
  %    Each output (feature) map is fully connected to all input maps
  
  properties
    ks; % [1]. kernel size
    M; % [1]. #output maps
    
    ker; 
    % [ks,ks,Min, Mout]. the kernels. 
    % Mi = #input maps, Mo = #output maps
    b; 
    % [Mout]. bias
    
    dker; % [ks,ks,Min, Mout]. the kernels derivatives. 
    db; % [Mout]. bias derivatives
    
    hpmker; % handle of parameter manager
    hpmb;
  end
  
  methods
    function obj = trans_conv_fc(ks_, M_)
      obj.ks = ks_;
      obj.M = M_;
      
      obj.hpmker = param_mgr_fmwl();
      obj.hpmb = param_mgr_fmwl();
    end
    
    function [obj, data_o] = ff(obj, data_i) 
    % 
      sz = size( data_i.a );
      data_o.a = zeros( [obj.szs_out(1:end-1), sz(end)] );
      Mout = obj.szs_out(3);
      for j = 1 : Mout
        % convolution:
        % a_in: [Hin,Win,Min, N]
        % the kernel: [ks,ks,Min, 1]
        % tmp: [Hou,Wou,1,N]
        % z = convn(a_in, obj.ker(:,:,:,j), 'valid'); % TODO: check this
        
        % convolution:
        sz = size(data_i.a);
        Min = sz(3);
        z2 = [];
        for i = 1 : Min
          tt = convn(squeeze(data_i.a(:,:,i,:)),...
                     obj.ker(:,:,i,j),  'valid');
          if (isempty(z2)), z2 = zeros( size(tt) ); end
          z2 = z2 + tt;
        end
        
        % no non-linear activation!
        data_o.a(:,:,j,:) = z2 + obj.b(j); % [Hou,Wou,1,N] + 1
      end % for j
    end % ff
    
    function data_i = deriv_input(obj, data_i, data_o)

      Mi = obj.szs_in(3); % assert(Mi == size(data_i.d,3));
      Mo = obj.szs_out(3); % assert(Mo == size(data_o.a,3));
      N = size(data_o.d, 4); % assert(N == size(data_o.a,4));

      for i = 1 : Mi
        z = zeros( [obj.szs_in(1),obj.szs_in(2),N] );
        for j = 1 : Mo
          tmp_ker = rot180( obj.ker(:,:,i,j) );
          z = z + convn(...
            squeeze(data_o.d(:,:,j,:)), tmp_ker, 'full');
        end % for j
        data_i.d(:,:,i,:) = z; 
      end % for i
    end % deriv_input
    
    function obj = deriv_param(obj, data_i, data_o)
      Mo = size(data_o.a, 3); % assert(Mo == size(data_o.d,3);
      Mi = size(data_i.a, 3);
      N = size(data_o.d, 4);
      Nden = 1/N;
      
      obj.db = zeros(Mo,1);
      for j = 1 : Mo
        %%% calculate dker
        for i = 1 : Mi
          % TODO: check this!
%           obj.dker(:,:,i,j) = Nden .* ...
%             rot180(convn(squeeze(data_i.a(:,:,i,:)),...
%                          rot180(squeeze(dxo(:,:,j,:))),...
%                          'valid') );
          obj.dker(:,:,i,j) = Nden .* ...
            convn(flipall(data_i.a(:,:,i,:)), data_o.d(:,:,j,:), 'valid');
        end % for i
        
        %%% calculate db
        tmp = data_o.d(:,:,j,:);
        obj.db(j) = sum(tmp(:)) ./ N;
      end % for j
    end % deriv_param
    
    function obj = update_param(obj, t)
      [obj.hpmker, obj.ker] = obj.hpmker.update_param(...
        obj.ker, obj.dker, t);
      [obj.hpmb, obj.b] = obj.hpmb.update_param(...
        obj.b, obj.db, t);
    end
    
    function obj = init_param(obj, szs)
      
      % set input map size
      obj.szs_in = szs;
      
      % randomly initialize the kernels
      Min = szs(3);
      Mout = obj.M;
      f = 0.01;
      obj.ker = f * randn( [obj.ks,obj.ks,Min, Mout] ); 

%       Min = szs(3);
%       Mout = obj.M;
%       obj.ker = 2*(rand(obj.ks,obj.ks,Min, Mout) - 0.5); % in range [-1,+1]
%       fan_in = obj.ks*obj.ks*Min;
%       fan_out = Mout;
%       obj.ker = obj.ker * sqrt(6/(fan_in + fan_out));

      % set zeros the bias
      obj.b = zeros(Mout,1);
      
      % deduce the output map size
      tmp = [szs(1), szs(2)];
      N = szs(4);
      obj.szs_out = ...
        [tmp(1)-obj.ks+1, tmp(2)-obj.ks+1, Mout, N];
    end
    
  end % methods
  
end