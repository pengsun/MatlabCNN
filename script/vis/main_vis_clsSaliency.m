%% load model
tmp = load('mo.mat', 'h','mu');
h = tmp.h;
mu = tmp.mu;
clear tmp;
%% data
load mnist_uint8;
train_x = double(reshape(train_x',28,28,60000))/255;
train_y = double(train_y');
test_x = double(reshape(test_x',28,28,10000))/255;
test_y = double(test_y');
K = size(train_y,1);
%% the image to be visualized and zero mean
idx = 16;
I = test_x(:,:,idx);
y = test_y(:,idx);

% mean image
I = I - mu;
%% the first-order Taylor expansion of Sc(I) at I
hvis = vis_cls();
W = hvis.calc_cls_saliency(h,I,y);
M = abs(W);
%% show
imtool(I', 'InitialMagnification',800);
Mmax = max(M(:)); Mmin = min(M(:));
MM = (M - Mmin)./(Mmax-Mmin);
imtool(MM', 'InitialMagnification',800);