clear all
clc
rng(1); % Initialize the random seed

patchWidth = 29;
nClasses = 138;
fileList = listMICCAI('./');

%% Training Data

nVoxels = 100000;
nPatchPerVoxel = 1;
pickVoxelFUN = @pickVxBalanced;
pickOrFUN = @pickOrOrthogonal;

[samples, targets, voxels, orientations] = extractPatches(fileList, ...
    nClasses, patchWidth, nVoxels, nPatchPerVoxel, ...
    pickVoxelFUN, pickOrFUN,...
    true, 'training.h5');

%% Testing Data

nVoxels = 2000;
nPatchPerVoxel = 3;
pickVoxelFUN = @pickVxRandomly;
pickOrFUN = @pickOrOrthogonal;

extractPatches(fileList, ...
    nClasses, patchWidth, nVoxels, nPatchPerVoxel, ...
    pickVoxelFUN, pickOrFUN, ...
    false, 'testing.h5');


