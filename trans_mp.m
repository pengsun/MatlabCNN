classdef trans_mp < trans_basic
  %TRANS_MP Max Pooling
  %   Use Jonathan Masci's implementation
  
  properties
    scale; % scale for subsampling
    idx; % index for the local maximum elements
  end
  
  methods
    function obj = trans_mp(scale_)
      obj.scale = scale_;
    end
    
    function [obj, data_o] = ff(obj, data_i) 
    %
      [data_o.a, obj.idx] = MaxPooling(data_i.a, [obj.scale,obj.scale]);
    end % ff    
    
    function data_i = deriv_input(obj, data_i, data_o)
    % data_i.d: [Hi,Wi,Mi,N]
    % data_o.d: [Ho,Wo,Mo,N], where [Hi,Wi] = s*[Ho,Wo]
      data_i.d  = zeros( obj.szs_in );
      data_i.d( obj.idx ) = data_o.d(:);
    end % deriv_input
    
    function obj = init_param(obj, szs_in_)
    % szs_in_: [a,b,c]. input map size
    % Set:
    %  obj.szs_out: [Hout,Wout,Mout]
    %  obj.szs_in: [Hin,Win,Min]
      
      % set input map size
      obj.szs_in = szs_in_;
      
      % deduce the output map size
      tmp = [szs_in_(1), szs_in_(2)];
      tmp = tmp ./ obj.scale;
      obj.szs_out = [tmp(1), tmp(2), szs_in_(3),szs_in_(4)];
    end % init_param
    
  end % methods
  
end