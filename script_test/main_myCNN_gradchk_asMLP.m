%% data
N = 21;
nb = 3;
X = rand(28,28,N);
Y = zeros(10,N);
for n = 1 : N
  ii = randi(10,1);
  Y(ii,n) = 1;
end

K = size(Y,1);
%% init
h = myCNN();

%%% layers
% fc
h.transArr{end+1} = trans_fc(40); 
% sigmoid
h.transArr{end+1} = trans_act_sigm(); 

% fc
h.transArr{end+1} = trans_fc(20);
% sigmoid
h.transArr{end+1} = trans_act_sigm(); 

% fc
h.transArr{end+1} = trans_fc(K);

%%% loss type
h.lossType = loss_softmax();

%%% other parameters
h.alpha = 0.5;
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
  
  h = update_param_grad(h);
end