function myCNN_gradchk(hcnn, x,y, varargin)
%MYCNN_GRADCHK Summary of this function goes here
%   Detailed explanation goes here
 
  if (nargin==3)
    st.epsilon = 1e-4;
    st.er = 1e-8;
  else
    st.epsilon = varargin{1};
    st.er = varargin{2};
  end
  
  
  st.h = hcnn;
  st.x = x;
  st.y = y;
  
  % check each layer
  L = numel( st.h.transArr );
  for ell = L : -1 : 1
    switch ( class(st.h.transArr{ell}) )
      case 'trans_fc'
        trans_fc_gradchk(st, ell);
%       case 'trans_fc_dropout'
%         trans_fc_gradchk(st, ell); % the same thing
      case 'trans_conv'
        trans_conv_gradchk(st, ell);
    end % switch
  end % for ell

end % myCNN_gradchk

function trans_fc_gradchk(st, ell)
  htrans = st.h.transArr{ell};
  
  %%% check db
  for j = 1 : numel(htrans.db)
    cnn_p = st.h;
    cnn_p.transArr{ell}.b(j) = cnn_p.transArr{ell}.b(j) + st.epsilon;
    cnn_p = cnn_p.ff(st.x);
    cnn_p = cnn_p.bp(st.y);
    
    cnn_m = st.h;
    cnn_m.transArr{ell}.b(j) = cnn_m.transArr{ell}.b(j) - st.epsilon;
    cnn_m = cnn_m.ff(st.x);
    cnn_m = cnn_m.bp(st.y);
    
    d = (cnn_p.L - cnn_m.L) ./ (2 * st.epsilon);
    e = abs( d - htrans.db(j) );
    assert( e <= st.er );
  end % for j
  
  %%% check dW
  for i1 = 1 : size(htrans.dW,1)
    for i2 = 1 : size(htrans.dW,2)
      for i3 = 1 : size(htrans.dW,3)
        cnn_p = st.h;
        cnn_p.transArr{ell}.W(i1,i2,i3) = ...
          cnn_p.transArr{ell}.W(i1,i2,i3) + st.epsilon;
        cnn_p = cnn_p.ff(st.x);
        cnn_p = cnn_p.bp(st.y);
        
        cnn_m = st.h;
        cnn_m.transArr{ell}.W(i1,i2,i3) =...
          cnn_m.transArr{ell}.W(i1,i2,i3) - st.epsilon;
        cnn_m = cnn_m.ff(st.x);
        cnn_m = cnn_m.bp(st.y);
        
        d = (cnn_p.L - cnn_m.L) ./ (2 * st.epsilon);
        e = abs( d - htrans.dW(i1,i2,i3) );
        assert( e <= st.er );
      end % for i3
    end % for i2
  end % for i1
  
end % trans_fc_gradchk

function trans_conv_gradchk(st, ell)
  htrans = st.h.transArr{ell};
  
  %%% check db
  for j = 1 : numel(htrans.b) % for output map
    cnn_p = st.h;
    cnn_p.transArr{ell}.b(j) = cnn_p.transArr{ell}.b(j) + st.epsilon;
    cnn_p = cnn_p.ff(st.x);
    cnn_p = cnn_p.bp(st.y);
    
    cnn_m = st.h;
    cnn_m.transArr{ell}.b(j) = cnn_m.transArr{ell}.b(j) - st.epsilon;
    cnn_m = cnn_m.ff(st.x);
    cnn_m = cnn_m.bp(st.y);
    
    d = (cnn_p.L - cnn_m.L) ./ (2 * st.epsilon);
    e = abs( d - htrans.db(j) );
    assert( e <= st.er );
    
  
    %%% check dker
    Mi = size(htrans.ker,3);
    for i = 1 : Mi % for input map
      
      for u = 1 : size(htrans.ker, 1)
        for v = 1 : size(htrans.ker, 2)
          cnn_p = st.h;
          cnn_p.transArr{ell}.ker(u,v,i,j) = ...
            cnn_p.transArr{ell}.ker(u,v,i,j) + st.epsilon;
          cnn_p = cnn_p.ff(st.x);
          cnn_p = cnn_p.bp(st.y);
          
          cnn_m = st.h;
          cnn_m.transArr{ell}.ker(u,v,i,j) = ...
            cnn_m.transArr{ell}.ker(u,v,i,j) - st.epsilon;
          cnn_m = cnn_m.ff(st.x);
          cnn_m = cnn_m.bp(st.y);
          
          d = (cnn_p.L - cnn_m.L) ./ (2 * st.epsilon);
          e = abs( d - htrans.dker(u,v,i,j) );
          assert( e <= st.er );
        end % for v
      end % for u
      
    end % for i input map
  end % for j output map
  
end % trans_conv_gradchk

