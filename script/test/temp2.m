%%
a = [5 4 3 2 1];
% a = ones(1,5);
% a = repmat(a, 2,1);
b = [5 4 3];
% b = repmat(b, 2,1);
%% direct
y = convn(a,b, 'valid');
%% sum of pulse
z = zeros(size(a)-size(b)+1);
for i = 1 : numel(a)
  aa = zeros( size(a) );
  aa(i) = a(i);
  t = convn(aa,b, 'valid')
  z = z + t;
end
z;


