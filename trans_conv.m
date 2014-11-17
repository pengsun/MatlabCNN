classdef trans_conv < trans_basic
  %TRANS_CONV Convolution
  %   Detailed explanation goes here
  
  properties
    ks; % [1]. kernel size
    M; % [1]. #output maps
    MM; % [1]. #input maps connected for each output map
    
    ker; 
    % [ks,ks,MM, Mo]. the kernels. 
    % Mi = #input maps, Mo = #output maps
    b; 
    % [Mout]. bias
    indMask; 
    % [Mi, Mo]. Indicator for the input-output map connection.
    % Each colum has MM 1s with the other elements 0.
    
    dker; % [ks,ks,Min, Mout]. the kernels derivatives. 
    db; % [Mout]. bias derivatives
    
    hpmker; % handle of parameter manager
    hpmb;
  end
  
  methods
    function obj = trans_conv(ks_, M_, varargin)
      obj.ks = ks_;
      obj.M = M_;
      obj.MM = -1;
      if ( ~isempty(varargin) ), obj.MM = varargin{1}; end
      
      obj.hpmker = param_mgr_fmwl();
      obj.hpmb = param_mgr_fmwl();
    end
    
    function [obj, data_o] = ff(obj, data_i, data_o) 
    % 
      N = data_i.N;
      data_o.a = zeros( [obj.szs_out(1:end-1), N] );
      Mout = obj.szs_out(3);
      Min = obj.szs_in(3);
      for j = 1 : Mout       
        % convolution: for each connected input map
        z2 = [];
        ii = 0; % sub-index for input map connected to j
        for i = 1 : Min
          if ( ~obj.indMask(i,j) ), continue; end % not connected, skip
          
          ii = ii + 1; % input map index
          tt = convn(squeeze(data_i.a(:,:,i,:)),... % note: i
                     obj.ker(:,:,ii,j),...          % note: ii
                     'valid');
          if (isempty(z2)), z2 = zeros( size(tt) ); end
          z2 = z2 + tt;
        end % for i
        
        % no non-linear activation!
        data_o.a(:,:,j,:) = z2 + obj.b(j); % [Hou,Wou,1,N] + 1
      end % for j
    end % ff
    
    function data_i = deriv_input(obj, data_i, data_o)

      Mi = obj.szs_in(3); % assert(Mi == size(data_i.d,3));
      Mo = obj.szs_out(3); % assert(Mo == size(data_o.a,3));
      N = data_o.N; % assert(N == size(data_o.a,4));

      for i = 1 : Mi
        z = zeros( [obj.szs_in(1),obj.szs_in(2),N] );

        for j = 1 : Mo
          if ( ~obj.indMask(i,j) ), continue; end % not connected, skip
          
          ii = sum( obj.indMask(1:i, j) ); % TODO: a reverse index be better
          z = z + convn(squeeze( data_o.d(:,:,j,:) ), ... % note: j
                        rot180( obj.ker(:,:,ii,j) ), ...  % note: ii
                       'full');
        end % for j
        data_i.d(:,:,i,:) = z; % note: i
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
        ii = 0;
        for i = 1 : Mi
          % TODO: check this!
%           obj.dker(:,:,i,j) = Nden .* ...
%             rot180(convn(squeeze(data_i.a(:,:,i,:)),...
%                          rot180(squeeze(dxo(:,:,j,:))),...
%                          'valid') );

          if ( ~obj.indMask(i,j) ), continue; end % not connected, skip
          ii = ii + 1; % input map index
          obj.dker(:,:,ii,j) = Nden .* ...       % note: ii
            convn(flipall(data_i.a(:,:,i,:)),... % note: i
                  data_o.d(:,:,j,:), ...         % note: j
                  'valid');
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
      Mi = szs(3);
      
      % index mask for connected input maps
      if (obj.MM<0), obj.MM = szs(3); end % unset; set it fully connected
      if (obj.MM>Mi), obj.MM = Mi; end % no more than #input maps
      obj.indMask = false(obj.MM, obj.M);
      for i = 1 : obj.M
        tmp = randsample(Mi, obj.MM);
        obj.indMask(tmp, i) = true;
      end
      
      % randomly initialize the kernels
      f = 0.01;
      obj.ker = f * randn( [obj.ks, obj.ks, obj.MM, obj.M] ); 

%       Min = szs(3);
%       Mout = obj.M;
%       obj.ker = 2*(rand(obj.ks,obj.ks,Min, Mout) - 0.5); % in range [-1,+1]
%       fan_in = obj.ks*obj.ks*Min;
%       fan_out = Mout;
%       obj.ker = obj.ker * sqrt(6/(fan_in + fan_out));

      % set zeros the bias
      obj.b = zeros(obj.M, 1);
      
      % deduce the output map size
      tmp = [szs(1), szs(2)];
      N = szs(4);
      obj.szs_out = ...
        [tmp(1)-obj.ks+1, tmp(2)-obj.ks+1, obj.M, N];
    end
    
  end % methods
  
end