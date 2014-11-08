classdef batchPart
  %BATCHPART Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    N; % #instances
    nb; % #batches
    
    
    kk; % random permutation, ineternal
    bs; % batch size
  end
  
  methods
    function obj = batchPart(N, nb)
      obj.N = N;
      obj.nb = floor(nb);
      
      obj.kk = randperm(N);
      obj.bs = floor(N/nb);
    end
    
    function ind = get_ind_from_batch(obj, ib)
      iStart = (ib-1)*obj.bs + 1;
      iEnd   = (ib-1)*obj.bs + obj.bs;
      ind = obj.kk(iStart : iEnd);
    end
  end
  
end

