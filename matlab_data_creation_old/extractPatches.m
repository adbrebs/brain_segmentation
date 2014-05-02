function [samples, targets, voxels, orientations] = extractPatches(...
    fileList, ...
    nClasses, patchWidth, nVoxels, nPatchPerVoxel, ...
    pickVoxelsFUN, pickPatchFUN, pickTargetFUN, ...
    permute, fileName)
% patchWidth needs to be odd to get a central voxel

tic

nFiles = length(fileList); 

% Adjust number of voxels in order to have the same number of images
% par file and per class
divisor = (nFiles * nClasses);
nVoxels = ceil(nVoxels / divisor) * divisor;
nVPerFile = nVoxels / nFiles;
nPatches = nVoxels * nPatchPerVoxel;
nSPerFile = nVPerFile * nPatchPerVoxel; % number of samples per file

% Initialize the containers that will store the samples for each file
% (required for parallelization)
samplesP = cell(1,nFiles);
patchLinIdxP = cell(1,nFiles);
targetsP = cell(1,nFiles);
voxelsP = cell(1,nFiles);
% normal vectors of the slices containing the patches
orientationsP = cell(1,nFiles);


for i = 1:nFiles

    % Open and crop the image to only keep the brain
    [mri, label] = openNII(fileList{i}, false);
    %[mri, label] = cropImg(mri, label);
    
    % Pick voxels
    [voxelsP{i}, vxLinIdx] = pickVoxelsFUN(label.img, nClasses, nVPerFile, nPatchPerVoxel);
    
    % Extract patches
    [orientationsP{i}, samplesP{i}, patchLinIdxP{i}] = pickPatchFUN(mri.img, ...
        voxelsP{i}, patchWidth);
    
    % Add labels to the patches
    targetsP{i} = pickTargetFUN(label.img, vxLinIdx, patchLinIdxP{i}, nClasses);

end

% Aggregate the data (required for parallelization)
samples = zeros(nPatches, patchWidth^2);
patchLinIdx = zeros(nPatches, patchWidth^2);
targets = zeros(nPatches, nClasses);
voxels = zeros(nPatches, 3);
orientations = zeros(nPatches, 3);
for i = 1:nFiles
    idxs = 1+(i-1)*nSPerFile:i*nSPerFile;
    samples(idxs,:) = samplesP{i};
    patchLinIdx(idxs,:) = patchLinIdxP{i};
    targets(idxs,:) = targetsP{i};
    voxels(idxs,:) = voxelsP{i};
    orientations(idxs,:) = orientationsP{i};
end
clear samplesP linIdxP targetsP pointsP orientationsP

% Permute the data
if permute
    pe = randperm(nSPerFile*nFiles);
    samples = samples(pe,:);
    patchLinIdx = patchLinIdx(pe,:);
    targets = targets(pe,:);
    voxels = voxels(pe,:);
    orientations = orientations(pe,:);
end

fileName = ['../data/' fileName];
% Save the data on the disk
h5create(fileName, '/patches', size(samples))
h5create(fileName, '/targets', size(targets))
h5create(fileName, '/voxels', size(voxels))
h5create(fileName, '/orientations', size(orientations))
h5write(fileName, '/patches', samples)
h5write(fileName, '/targets', targets)
h5write(fileName, '/voxels', voxels)
h5write(fileName, '/orientations', orientations)

% attributes
h5writeatt(fileName,'/','creation_date',datestr(now));
h5writeatt(fileName,'/','n_voxels', nVoxels);
h5writeatt(fileName,'/','n_patch_per_voxel', nPatchPerVoxel);
h5writeatt(fileName,'/','n_patches', nPatches);
h5writeatt(fileName,'/','patch_width', patchWidth);
h5writeatt(fileName,'/','n_classes', nClasses);

toc

end

