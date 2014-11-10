%% matrix
x = 1:25;
x = reshape(x, 5,5);
x = repmat(x, [1,1,7]);
%% rearrage
xp = permute(x, [3,1,2]);
xr = ipermute(xp, [3,1,2]);
%% conv
tmpl = ones(3,1)/3;
yp = convn(xp,tmpl, 'same');
%% rearrage
y = ipermute(yp, [3,1,2]);