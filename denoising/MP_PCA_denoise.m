function [denoised,S_before,S_after,P] = MP_PCA_denoise(image,window,mask)
% Marchenko-Pastur Principal Component Analysis (MP-PCA) denoising
% J. Veraart, "Denoising of diffusion MRI using random matrix theory"
% Modified from the original algorithm as found on Sune Jespersen's GitHub page: https://github.com/sunenj 
%
% input:
% image:  images to be denoised. Must have 3 or 4 indices with MRI images 
%         along the last index and voxels in the first 2 or 3.
% window: sliding window
% mask:   is true for all voxels per default but can be manually set to mask out regions.
%
% output:
% denoised: denoised images
% S_before: map of estimated noise standard deviation variance before denoising
% S_after:  map of estimated noise standard deviation variance after denoising
% P:        number of detected signal principal components

%% adjust image dimensions and assert
dimsOld = size(image);
if ~exist('mask','var')
    mask = [];
end
[image,mask] = MP_PCA_imageAssert(image,mask);

dims = size(image);
assert(length(window)>1 && length(window)<4,'window must have 2 or 3 dimensions')
assert(all(window>0),'window values must be strictly positive')
assert(all(window<=dims(1:length(window))),'window values must not exceed image dimensions')
if length(window)==2
    window(3) = 1;
end

%% denoise image
denoised = zeros(size(image));
P = zeros(dims(1:3));
S2_before = zeros(dims(1:3));
S2_after = zeros(dims(1:3));
M = dims(1)-window(1)+1;
N = dims(2)-window(2)+1;
O = dims(3)-window(3)+1;
count = zeros(dims(1:3));
for index = 0:M*N*O-1
    k = 1 + floor(index/M/N);
    j = 1 + floor(mod(index,M*N)/M);
    i = 1 + mod(mod(index,M*N),M);
    rows = i:i-1+window(1);
    cols = j:j-1+window(2);
    slis = k:k-1+window(3);

    % Create X data matrix
    X = reshape(image(rows,cols,slis,:),[],dims(4))';
    
    % remove masked out voxels
    maskX = reshape(mask(rows,cols,slis),[],1)';
    if nnz(maskX)==0 || nnz(maskX)==1
        continue
    end
    maskX = logical(maskX);
    
    % denoise X
    [X(:,maskX),s2_before,s2_after,p] = MP_PCA_denoiseMatrix(X(:,maskX));

    % assign
    X(:,~maskX) = 0;
    denoised(rows,cols,slis,:) = denoised(rows,cols,slis,:) + reshape(X',[window dims(4)]);
    P(rows,cols,slis) = P(rows,cols,slis) + p;
    S2_before(rows,cols,slis) = S2_before(rows,cols,slis) + s2_before;
    S2_after(rows,cols,slis) = S2_after(rows,cols,slis) + s2_after;
    count(rows,cols,slis) = count(rows,cols,slis) + 1;
end

skipped = count==0 | ~mask;
denoised = denoised + image.*skipped; % Assign original data to denoisedImage outside of mask and at skipped voxels
count(count==0) = 1;
denoised = denoised./count;
S2_before = S2_before./count;
S2_after = S2_after./count;
P = P./count;

S2_before(~mask) = nan;
S2_after(~mask) = nan;
P(~mask) = nan;

%% adjust output to match input dimensions
denoised = reshape(denoised, dimsOld);
P = reshape(P,dimsOld(1:end-1));
S2_before = reshape(S2_before,dimsOld(1:end-1));
S2_after = reshape(S2_after,dimsOld(1:end-1));
S_before = sqrt(S2_before);
S_after = sqrt(S2_after);

end

