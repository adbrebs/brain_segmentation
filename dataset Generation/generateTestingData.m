
rng(1); % Initialize the random seed

nVoxels = 10000;
patchWidth = 29;
nPatchPerVoxel = 10;
nClasses = 138;


[samples, targets, voxels, orientations] = extractPatches(...
    nClasses, patchWidth, nVoxels, nPatchPerVoxel, ...
    @pickVoxelsRandomly, false, true);


%% Read the data
% fileName = './../data/mridata.h5';
% samples = h5read(fileName, '/inputs');
% 
% % Display one patch
% slice = reshape(samples(4,:), patchWidth, patchWidth);
% 
% figure
% slice = extractSlice(mri.img,100,100,100,100,100,100,30);
% colormap(gray)
% imagesc(slice) 
