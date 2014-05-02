
clc
rng(1); % Initialize the random seed

patchWidth = 29;
nClasses = 138;
fileList = listMICCAI('./');

%% Training Data

nVoxels = 100000;
nPatchPerVoxel = 1;
pickVoxelFUN = pickVxFactory('inPlane');
pickPatchFUN = pickPatchFactory('parallelXZ');
pickTargetFUN = pickTargetFactory('centered');

[samples, targets, voxels, orientations] = extractPatches(fileList, ...
    nClasses, patchWidth, nVoxels, nPatchPerVoxel, ...
    pickVoxelFUN, pickPatchFUN, pickTargetFUN, ...
    true, 'training1.h5');

%% Testing Data

nVoxels = 2000;
nPatchPerVoxel = 1;
pickVoxelFUN = pickVxFactory('inPlane');
pickPatchFUN = pickPatchFactory('parallelXZ');
pickTargetFUN = pickTargetFactory('centered');

extractPatches(fileList, ...
    nClasses, patchWidth, nVoxels, nPatchPerVoxel, ...
    pickVoxelFUN, pickPatchFUN, pickTargetFUN, ...
    false, 'testing1.h5');


