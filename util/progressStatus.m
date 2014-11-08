classdef progressStatus
  %PROSTATUS Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    N; % total number
    nc; % #check point
  end
  
  properties
    len; % length between two check points
    chkpnt; % check point
    ichkpnt; % index (count) of check point
  end
  
  methods
    function obj = progressStatus(N_, nc_)
      obj.N = N_;
      obj.nc = nc_;
      
      obj.len = floor(N_/nc_);
      obj.chkpnt = obj.len;
      obj.ichkpnt = 1;
    end
    
    function [obj, str] = testChkPnt(obj, count)
      str = '';
      if (count>=obj.chkpnt)
        str = sprintf('%d%%', round(100/obj.nc)*obj.ichkpnt);
        obj.chkpnt = obj.chkpnt + obj.len;
        obj.ichkpnt = obj.ichkpnt + 1;
      end
      
    end % testChkPnt
  end
  
end

