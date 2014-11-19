%%
X = 512;
Z = 300;
N = 1;
k = 5;

% sz = [X,X,Z, N];
% sztmpl = [k,k,k];
sz = [X,X, N];
sztmpl = [k,k];

n = 10000;
%%
gc=convn(gpuArray.rand(sz,'single'),gpuArray.rand(sztmpl,'single'));
tic;
for i=1:n
    gc=convn(gpuArray.rand(sz,'single'),gpuArray.rand(sztmpl,'single'));
end
toc
%%
c=convn(rand(sz,'single'),rand(sztmpl,'single'));
tic;
for i=1:n
    c=convn(rand(sz,'single'),rand(sztmpl,'single'));
end
toc