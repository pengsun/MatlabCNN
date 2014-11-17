%% load model
tmp = load('mo.mat', 'h','mu');
h = tmp.h;
mu = tmp.mu;
clear tmp;
%% data
% load mnist_uint8;
% train_x = double(reshape(train_x',28,28,60000))/255;
% train_y = double(train_y');
% test_x = double(reshape(test_x',28,28,10000))/255;
% test_y = double(test_y');
% K = size(train_y,1);
%% the class to be visualized and zero mean
c = 5;
K = 10;
y = zeros(K,1);
y(c) = 1;
%% the class model
hvis = vis_cls();
[I,s] = hvis.calc_cls_model(h, y, mu);
%% show
imtool(I', [], 'InitialMagnification',800);
% Mmax = max(M(:)); Mmin = min(M(:));
% MM = (M - Mmin)./(Mmax-Mmin);
% imtool(MM', 'InitialMagnification',800);
%% print
fprintf('c = %d\n', c);
fprintf('class score:\n');
disp(s);