classdef trans_conv < trans_basic
  %TRANS_CONV Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function obj = trans_conv(ks_, M_)
      obj = trans_conv_impl(ks_, M_);
    end
    
    
  end % methods
  
end