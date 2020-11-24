function [X,s2_before,s2_after,p] = MP_PCA_denoiseMatrix(X) 
% X: denoised matrix
% s2: original noise variance
% s2_after: noise variance after denoising
% p: number of signal components

M = size(X,1);
N = size(X,2);
if M<N
    [U,lambda] = eig(X*X','vector');
else
    [U,lambda] = eig(X'*X,'vector');
end
[lambda,order] = sort(lambda,'descend');
U = U(:,order);
csum = cumsum(lambda,'reverse');
p = (0:length(lambda)-1)';
p = -1 + find((lambda-lambda(end)).*(M-p).*(N-p) < 4*csum*sqrt(M*N),1);
if p==0
    X = zeros(size(X));
elseif isempty(p) % Occurs in voxels with precisely 0 signal
    X = zeros(size(X));
    p = 0;
elseif M<N
    X = U(:,1:p)*U(:,1:p)'*X;
else
    X = X*U(:,1:p)*U(:,1:p)';
end
s2_before = csum(p+1)/((M-p)*(N-p));
s2_after = s2_before - csum(p+1)/(M*N);
end