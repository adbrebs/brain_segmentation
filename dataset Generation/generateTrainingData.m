
rng(1); % Initialize the random seed

nVoxels = 100000;
nPatchPerVoxel = 1;
patchWidth = 29;
nClasses = 138;


[samples, targets, voxels, orientations] = extractPatches(...
    nClasses, patchWidth, nVoxels, nPatchPerVoxel, ...
    @pickBalancedVoxels, true, true);


%% Read the data
fileName = './../data/mridata.h5';
samples = h5read(fileName, '/inputs');

% Display one patch
slice = reshape(samples(4,:), patchWidth, patchWidth);

figure
slice = extractSlice(mri.img,100,100,100,100,100,100,30);
colormap(gray)
imagesc(slice) 
