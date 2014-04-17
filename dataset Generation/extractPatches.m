function [samples, targets, voxels, orientations] = extractPatches(...
    nClasses, patchWidth, ...
    nVoxels, nPatchPerVoxel, pickVoxelsFun, permute, writeFile)
% patchWidth needs to be odd to get a central voxel

tic

mriFolder = './mri/';

files = dir([mriFolder '*.nii']);
nFiles = length(files);


divisor = (nFiles * nClasses);
nVoxels = ceil(nVoxels / divisor) * divisor;
nVPerFile = ceil(nVoxels / nFiles);
nSamples = nVoxels * nPatchPerVoxel;
nSPerFile = nSamples / nFiles; % number of samples per file

% Initialize the containers that will store the samples for each file
% (required for parallelization)
samplesP = cell(1,nFiles);
targetsP = cell(1,nFiles);
voxelsP = cell(1,nFiles);
% normal vectors of the slices containing the patches
orientationsP = cell(1,nFiles);


parfor i = 1:nFiles

    samplesP{i} = zeros(nSPerFile, patchWidth^2);
    targetsP{i} = zeros(nSPerFile, 1);
    voxelsP{i} = zeros(nSPerFile, 3);
    orientationsP{i} = zeros(nSPerFile, 3);
    
    % Open and crop the image to only keep the brain
    [mri, label] = openNII(files(i).name);
    [mri, label] = cropImg(mri, label);
    
    dims = size(mri.img);
    
    % Select random voxels inside the brain and label the future patch
    % linIdx = pickVoxelsRandomly(label.img, nSPerFile);
    linIdx = pickVoxelsFun(label.img, nClasses, nVPerFile);
    [x, y, z] = ind2sub(dims, linIdx);
    
    for j = 1:nVPerFile
        
        for k = 1:nPatchPerVoxel
        
            idx = k + (j-1) * nPatchPerVoxel;
            % Generate a random orientation and extract the patch from the
            % corresponding slice
            or = rand(1,3);
            buff = extractSlice(mri.img,x(j),y(j),z(j),or(1),or(2),or(3),...
                floor(patchWidth / 2));
            samplesP{i}(idx,:) = buff(:);
            orientationsP{i}(idx,:) = or;
            targetsP{i}(idx) = label.img(linIdx(j));
            voxelsP{i}(idx,:) = [x(j), y(j), z(j)];
            
        end
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

if (writeFile)
    fileName = ['./../data/mridata_', num2str(patchWidth), '_',...
    num2str(nSamples), '_', datestr(now) '.h5'];

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
    h5writeatt(fileName,'/','n_PatchPerVoxel', nPatchPerVoxel);
    h5writeatt(fileName,'/','n_Samples', nSamples);
    h5writeatt(fileName,'/','patch_width', patchWidth);
    h5writeatt(fileName,'/','n_classes', nClasses);
end

toc

end

