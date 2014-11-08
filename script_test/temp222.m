%%
a = 25:-1:1;
a = reshape(a, 5,5);

% c = [9 8 7;6 5 4;3 2 1];
c = ones(4,4);
% b = repmat(b, 2,1);
%% direct
y = convn(a,c, 'valid');
%% sum of pulse
z = [];
for i = 1 : size(c,1)
  for j = 1 : size(c,2)
    cc = zeros( size(c) );
    cc(i,j) = c(i,j);
    t = convn(a,cc, 'valid')
    if (isempty(z)), z = zeros(size(t)); end
    z = z + t;
  end
end
% z;