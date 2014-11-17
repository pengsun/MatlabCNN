%% data
N = 27;
nb = 3;
K = 10;
[X,Y] = make_rand_inst(14,14, K, N);
%% init
h = myCNN();

%%% layers
% convolution, kernel size 5, #output map = 7
h.transArr{end+1} = trans_conv(5, 7); 
% activation
h.transArr{end+1} = trans_act_sigm(); 
% contrast normalization
h.transArr{end+1} = trans_respnorm();

% subsample, scale 2
h.transArr{end+1} = trans_sub(2); 

% convolution, kernel size 5, #output map = 8
h.transArr{end+1} = trans_conv(4, 10); 
% activation
h.transArr{end+1} = trans_act_sigm(); 
% contrast normalization
h.transArr{end+1} = trans_respnorm();

% subsample, scale 2
h.transArr{end+1} = trans_sub(2); 

% full connection, #output map = #classes
h.transArr{end+1} = trans_fc(K);


%%% loss type
h.lossType = loss_le();

%%% other parameters
h.batchsize = N/nb;
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
  
  h = init_dataArr(h, N/nb);
  h = ff(h, batch_x);
  h = bp(h, batch_y);
  myCNN_gradchk(h, batch_x, batch_y, 1e-4, 1e-5);
  fprintf('Congratulations: gradient checking done\n');
  
%   h = update_param(h, j);
end