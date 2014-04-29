
clc
rng(1); % Initialize the random seed

patchWidth = 29;
nClasses = 138;
fileList = listMICCAI('./');

%% Training Data

nVoxels = 10000;
nPatchPerVoxel = 1;
pickVoxelFUN = pickVxFactory('balanced');
pickPatchFUN = pickPatchFactory('parallelXZ');
pickTargetFUN = pickTargetFactory('proportion');

[samples, targets, voxels, orientations] = extractPatches(fileList, ...
    nClasses, patchWidth, nVoxels, nPatchPerVoxel, ...
    pickVoxelFUN, pickPatchFUN, pickTargetFUN, ...
    true, 'training_par_tar.h5');

%% Testing Data

nVoxels = 2000;
nPatchPerVoxel = 1;
pickVoxelFUN = pickVxFactory('random');
pickPatchFUN = pickPatchFactory('parallelXZ');
pickTargetFUN = pickTargetFactory('proportion');

extractPatches(fileList, ...
    nClasses, patchWidth, nVoxels, nPatchPerVoxel, ...
    pickVoxelFUN, pickPatchFUN, pickTargetFUN, ...
    false, 'testing_par_tar.h5');


