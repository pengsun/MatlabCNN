%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error
%% data
load mnist_uint8;

train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');
K = size(train_y,1);
%%
% rand('state',0);
% tr_ind = randsample(60000, 20000);
% train_x = train_x(:,:, tr_ind);
% train_y = train_y(:, tr_ind);
% te_ind = randsample(10000, 2000);
% test_x = test_x(:,:, te_ind);
% test_y = test_y(:, te_ind);
%% Image Mean Subtraction
tmp = cat(3, train_x, test_x);
mu = mean(tmp, 3);

train_x = bsxfun(@minus, train_x, mu);
test_x = bsxfun(@minus, test_x, mu);
%% init
h = myCNN();

%%% layers
% % dropout for input
% h.transArr{end+1} = trans_act_dropout(0.8); 

% convolution, kernel size 5, #output map = 6
h.transArr{end+1} = trans_conv(5, 6); 
h.transArr{end}.hpmker = param_mgr_naive();
h.transArr{end}.hpmb = param_mgr_naive();
% activation
% h.transArr{end+1} = trans_act_relu();

% subsample, scale 2
h.transArr{end+1} = trans_sub(2); 
% activation
% h.transArr{end+1} = trans_act_relu();

% convolution, kernel size 5, #output map = 12
h.transArr{end+1} = trans_conv(5, 12);
h.transArr{end}.hpmker = param_mgr_naive();
h.transArr{end}.hpmb = param_mgr_naive();
% activation
% h.transArr{end+1} = trans_act_relu();

% subsample, scale 2
h.transArr{end+1} = trans_sub(2);
% activation
% h.transArr{end+1} = trans_act_relu();

% full connection, #output map = 100
h.transArr{end+1} = trans_fc(100);
h.transArr{end}.hpmW = param_mgr_naive();
h.transArr{end}.hpmb = param_mgr_naive();
% activation
h.transArr{end+1} = trans_act_relu();

% full connection, #output map = #classes
h.transArr{end+1} = trans_fc(K);
h.transArr{end}.hpmW = param_mgr_naive();
h.transArr{end}.hpmb = param_mgr_naive();

%%% loss
h.lossType = loss_softmax();

%%% other parameters
h.batchsize = 50;
h.numepochs = 2;
%% train
rand('state',0);
h = h.train(train_x, train_y);
%% test
pre_y = h.test(test_x);
[~,pre_c] = max(pre_y);
[~,test_c] = max(test_y);
err = mean(pre_c ~= test_c);
fprintf('err = %d\n', err);
%% results
%plot mean squared error
figure; plot(h.rL);

% fprintf('err = %d\n',err);
% assert(err<0.12, 'Too big error');