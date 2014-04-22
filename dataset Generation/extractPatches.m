function [samples, targets, voxels, orientations] = extractPatches(...
    fileList, ...
    nClasses, patchWidth, nVoxels, nOrPerVoxel, ...
    pickVoxelsFun, pickOrientationsFUN, ...
    permute, fileName)
% patchWidth needs to be odd to get a central voxel

tic

nFiles = length(fileList); 

% Adjust number of voxels in order to have the same number of images
% par file and per class
divisor = (nFiles * nClasses);
nVoxels = ceil(nVoxels / divisor) * divisor;
nVPerFile = ceil(nVoxels / nFiles);
nSamples = nVoxels * nOrPerVoxel;
nSPerFile = nVPerFile * nOrPerVoxel; % number of samples per file

% Initialize the containers that will store the samples for each file
% (required for parallelization)
samplesP = cell(1,nFiles);
targetsP = cell(1,nFiles);
voxelsP = cell(1,nFiles);
% normal vectors of the slices containing the patches
orientationsP = cell(1,nFiles);


for i = 1:nFiles

    samplesP{i} = zeros(nSPerFile, patchWidth^2);
    
    % Open and crop the image to only keep the brain
    [mri, label] = openNII(fileList{i}, false);
    [mri, label] = cropImg(mri, label);
    
    % Select random voxels inside the brain and label the future patch
    % linIdx = pickVoxelsRandomly(label.img, nSPerFile);
    linIdx = pickVoxelsFun(label.img, nClasses, nVPerFile);
    linIdx = linIdx(ceil((1:nOrPerVoxel*size(linIdx,1))/nOrPerVoxel), :);
    [x, y, z] = ind2sub(size(mri.img), linIdx);
    voxelsP{i} = [x, y, z];
    targetsP{i} = label.img(linIdx);
    
    % Select orientations for the voxels
    orientationsP{i} = pickOrientationsFUN(nVPerFile, nOrPerVoxel);
    
    for j = 1:length(linIdx)
        vx = voxelsP{i}(j,:);
        or = orientationsP{i}(j,:);
        % Extract the patch
        buff = extractSlice(mri.img, vx(1), vx(2), vx(3), ...
            or(1), or(2), or(3), floor(patchWidth / 2));
        samplesP{i}(j,:) = buff(:);
    end
end

% Aggregate the data (required for parallelization)
samples = zeros(nSamples, patchWidth^2);
targets = zeros(nSamples, 1);
voxels = zeros(nSamples, 3);
orientations = zeros(nSamples, 3);
for i = 1:nFiles
    idxs = 1+(i-1)*nSPerFile:i*nSPerFile;
    samples(idxs,:) = samplesP{i};
    targets(idxs,:) = targetsP{i};
    voxels(idxs,:) = voxelsP{i};
    orientations(idxs,:) = orientationsP{i};
end
clear samplesP targetsP pointsP orientationsP

% Permute the data
if permute
    pe = randperm(nSPerFile*nFiles);
    samples = samples(pe,:);
    targets = targets(pe,:);
    voxels = voxels(pe,:);
    orientations = orientations(pe,:);
end

% Save the data on the disk
h5create(fileName, '/inputs', size(samples))
h5create(fileName, '/targets', size(targets))
h5create(fileName, '/points', size(voxels))
h5create(fileName, '/orientations', size(orientations))
h5write(fileName, '/inputs', samples)
h5write(fileName, '/targets', targets)
h5write(fileName, '/points', voxels)
h5write(fileName, '/orientations', orientations)

% attributes
h5writeatt(fileName,'/','creation_date',datestr(now));
h5writeatt(fileName,'/','n_voxels', nVoxels);
h5writeatt(fileName,'/','n_patch_per_voxel', nOrPerVoxel);
h5writeatt(fileName,'/','n_samples', nSamples);
h5writeatt(fileName,'/','patch_width', patchWidth);
h5writeatt(fileName,'/','n_classes', nClasses);

toc

end

