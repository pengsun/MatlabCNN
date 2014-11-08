%%
ain = rand(24,24,6, 7);
ker = rand(2,2)./4;
%% direct
b1 = convn(ain,ker);
%% plane by plane
for j = 1 : 6
  b2(:,:,j,:) = convn(ain(:,:,j,:),ker);
end
%%
all( b1(:)==b2(:) )