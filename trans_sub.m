classdef trans_sub < trans_basic
  %TRANS_SUB Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    scale; % scale for subsampling
  end
  
  methods
    function obj = trans_sub(scale_)
      obj.scale = scale_;
    end
    
    function [obj, data_o] = ff(obj, data_i, data_o) 
    %
      % first average
      avg_tmpl = ones(obj.scale,obj.scale);
      avg_tmpl = avg_tmpl./numel(avg_tmpl);
      tmp = convn(data_i.a, avg_tmpl, 'valid');
      
      % then subsample every obj.scale pixels
      data_o.a = tmp(1:obj.scale:end, 1:obj.scale:end, :,:);
    end % ff    
    
    function data_i = deriv_input(obj, data_i, data_o)
      s = obj.scale;
      % data_i.d: [Hi,Wi,Mi,N]
      % data_o.d: [Ho,Wo,Mo,N], where [Hi,Wi] = s*[Ho,Wo]
      data_i.d  = expand(data_o.d, [s,s,1,1]) ./ (s*s);
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