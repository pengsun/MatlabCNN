%% for MNIST. (Input=28x28)-800-800-(Output=10)
% 
% dropout:
%   0.5 for hidden layer, 0.8 for input layer
% Max-norm:
%   c = 2 for the incoming weight vector for each hidden unit
% SGD with momentum:
%   
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
%% init
h = myCNN();

% parameters
h.batchsize = 1;
h.numepochs = 2000;
cc = 2;

%%% layers
% dropout for input layer
h.transArr{end+1} = trans_act_dropout(0.8); 

% FC
h.transArr{end+1} = trans_fc(800); 
h.transArr{end}.c = cc;
% h.transArr{end}.hpmW = param_mgr_naive();
% h.transArr{end}.hpmb = param_mgr_naive();
h.transArr{end}.hpmW = param_mgr_momentum();
h.transArr{end}.hpmb = param_mgr_momentum();
% sigmoid
h.transArr{end+1} = trans_act_sigm(); 
% dropout
h.transArr{end+1} = trans_act_dropout(); 

% FC
h.transArr{end+1} = trans_fc(800); 
h.transArr{end}.c = cc;
% h.transArr{end}.hpmW = param_mgr_naive();
% h.transArr{end}.hpmb = param_mgr_naive();
h.transArr{end}.hpmW = param_mgr_momentum();
h.transArr{end}.hpmb = param_mgr_momentum();
% sigmoid
h.transArr{end+1} = trans_act_sigm(); 
% dropout
h.transArr{end+1} = trans_act_dropout(); 

% full connection, #output map = #classes
h.transArr{end+1} = trans_fc(K);
h.transArr{end}.c = cc;
% h.transArr{end}.hpmW = param_mgr_naive();
% h.transArr{end}.hpmb = param_mgr_naive();
h.transArr{end}.hpmW = param_mgr_momentum();
h.transArr{end}.hpmb = param_mgr_momentum();

%%% loss
h.lossType = loss_softmax();
%% train
h = h.train(train_x, train_y);
save('mo_tmp.mat', 'h');
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