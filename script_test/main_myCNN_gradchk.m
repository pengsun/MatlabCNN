%% data
N = 21;
nb = 3;
K = 10;
[X,Y] = make_rand_inst(28,28, K, N);
%% init
h = myCNN();

%%% layers
% convolution, kernel size 5, #output map = 2
h.transArr{end+1} = trans_conv(5, 2); 
% sigmoid
h.transArr{end+1} = trans_act_sigm(); 

% subsample, scale 2
h.transArr{end+1} = trans_sub(2); 

% convolution, kernel size 5, #output map = 3
h.transArr{end+1} = trans_conv(5, 3);
% sigmoid
h.transArr{end+1} = trans_act_sigm(); 

% subsample, scale 2
h.transArr{end+1} = trans_sub(2);

% convolution, kernel size 2, #output map = 4
h.transArr{end+1} = trans_conv(2, 4);
% sigmoid
h.transArr{end+1} = trans_act_sigm(); 

% full connection, #output map = #classes
h.transArr{end+1} = trans_fc(K);
% sigmoid
% h.transArr{end+1} = trans_act_sigm(); 

%%% loss type
h.lossType = loss_softmax();

%%% other parameters
h.batchsize = 50;
h.numepochs = 2;
%% 
bpart = batchPart(N,nb); % instance batches
for j = 1 : nb
  fprintf('Batch %d/%d: ',j, nb);
  
  ind = bpart.get_ind_from_batch(j);
  batch_x = X(:, :, ind);
  batch_y = Y(:,    ind);
  
  if (j==1)
    sz(1) = size(batch_x,1); 
    sz(2) = size(batch_x,2);
    sz(3) = 1;
    sz(4) = N/nb;
    h = init_trans_param(h, sz);
  end
  
  h = ff(h, batch_x);
  h = bp(h, batch_y);
  myCNN_gradchk(h, batch_x, batch_y);
  fprintf('Congratulations: gradient checking done\n');
  
%   h = update_param_grad(h);
end