
rng(1); % Initialize the random seed

nSamples = 30 * 35 * 138;
patchWidth = 29;
fileName = ['./../data/mridata_', num2str(patchWidth), '_',...
    num2str(nSamples), '_', date '.h5'];

mriFolder = './mri/';
labelFolder = './label/';
nClasses = 138;

radius = floor(patchWidth / 2);

files = dir([mriFolder '*.nii']);
nFiles = length(files);
nSPerFile = ceil(nSamples / nFiles); % number of samples per file

% Initialize the containers that will store the samples for each file
% (required for parallelization)
samplesP = cell(1,nFiles);
targetsP = cell(1,nFiles);
voxelsP = cell(1,nFiles);
% normal vectors of the slices containing the patches
orientationsP = cell(1,nFiles);


tic
parfor i = 1:nFiles

    samplesP{i} = zeros(nSPerFile, patchWidth^2);
    targetsP{i} = zeros(nSPerFile, 1);
    voxelsP{i} = zeros(nSPerFile, 3);
    orientationsP{i} = zeros(nSPerFile, 3);
    
    % Open and crop the image to only keep the brain
    [mri, label] = openNII(files(i).name);
    [mri, label] = cropImg(mri, label);
    mri.img(label.img == 0) = 0; % Only keep the brain
    
    dims = size(mri.img);
    
    % Select random voxels inside the brain and label the future patch
    % linIdx = pickVoxelsRandomly(label.img, nSPerFile);
    linIdx = pickBalancedVoxels(label.img, nClasses, nSPerFile);
    [x, y, z] = ind2sub(dims, linIdx);
    targetsP{i} = label.img(linIdx);
    voxelsP{i} = [x, y, z];
    
    for j = 1:nSPerFile
        % Generate a random orientation and extract the patch from the
        % corresponding slice
        or = rand(1,3);
        buff = extractSlice(mri.img,x(j),y(j),z(j),or(1),or(2),or(3),radius);
        samplesP{i}(j,:) = buff(:);
        orientationsP{i}(j,:) = or;
    end
end
toc

% Aggregate the data (required for parallelization)
samples = zeros(nSPerFile*nFiles, patchWidth^2);
targets = zeros(nSPerFile*nFiles, 1);
voxels = zeros(nSPerFile*nFiles, 3);
orientations = zeros(nSPerFile*nFiles, 3);
for i = 1:nFiles
    idxs = 1+(i-1)*nSPerFile:i*nSPerFile;
    samples(idxs,:) = samplesP{i};
    targets(idxs,:) = targetsP{i};
    voxels(idxs,:) = voxelsP{i};
    orientations(idxs,:) = orientationsP{i};
end
clear samplesP targetsP pointsP orientationsP

% Permute the data
pe = randperm(nSPerFile*nFiles);
samples = samples(pe,:);
targets = targets(pe,:);
voxels = voxels(pe,:);
orientations = orientations(pe,:);

%% Save the data on the disk
h5create(fileName, '/inputs', size(samples))
h5create(fileName, '/targets', size(targets))
h5create(fileName, '/points', size(voxels))
h5create(fileName, '/orientations', size(orientations))
h5write(fileName, '/inputs', samples)
h5write(fileName, '/targets', targets)
h5write(fileName, '/points', voxels)
h5write(fileName, '/orientations', orientations)



% %% Read the data
% fileName = './../data/mridata.h5';
% samples = h5read(fileName, '/inputs');
% 
% % Display one patch
% slice = reshape(samples(4,:), patchWidth, patchWidth);
% colormap(gray)
% imagesc(slice) 
