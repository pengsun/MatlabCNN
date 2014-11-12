%%
m = 512;
h = 300;
mm = 128;
N = 1;
k=3;

sz = [m,m,h,N];
sztmpl = [k,k,2];

n = 10;
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