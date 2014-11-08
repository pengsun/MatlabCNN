%%
% a = [5 4 3 2 1];
a = ones(5,5);
% a = 10*rand(5,5);
% a = repmat(a, 2,1);
b = [9 8 7;6 5 4;3 2 1];
% b = repmat(b, 2,1);
%% direct
y = convn(a,b, 'valid');
%% sum of pulse
z = [];
for i = 1 : size(a,1)
  for j = 1 : size(a,2)
    aa = zeros( size(a) );
    aa(i,j) = a(i,j);
    t = convn(aa,b, 'valid')
    if (isempty(z)), z = zeros(size(t)); end
    z = z + t;
  end
end
% z;