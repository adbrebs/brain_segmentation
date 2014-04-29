function pickPatchFUN = pickPatchFactory(mode)

switch mode
    case 'orthogonal'
        pickPatchFUN = @pickPatchOrthogonal;
    case 'random'
        pickPatchFUN = @pickPatchRandomly;
    case 'parallelXZ'
        pickPatchFUN = @pickPatchParallelXZ;
    otherwise
        disp('Error choice pickTargetFUN')
end

end



function [orientations, samples] = pickPatchOrthogonal(...
    img, voxels, patchWidth)
% TODO: write this code


if nOrPerVoxel == 1
    or = zeros(nVPerFile,3);
    plan = randi(3,1,nVPerFile);
    or(sub2ind(size(or), 1:length(plan), plan)) = 1;
elseif nOrPerVoxel == 3
    or = repmat(diag(ones(3,1)), nVPerFile, 1);
else
    disp('this function only works for nOrPerVoxel = 1 or 3');
end


end


function [orientations, samples, linIdx] = pickPatchRandomly(...
    img, voxels, patchWidth)

nSPerFile = size(voxels,1);
radius = floor(patchWidth / 2);

samples = zeros(nSPerFile, patchWidth^2);
linIdx = zeros(nSPerFile, patchWidth^2);
orientations = rand(nSPerFile,3);

for j = 1:nSPerFile

    vx = voxels(j,:);
    or = orientations(j,:);
    
    % Extract the patch
    [buff1, buff2] = extractSlice(img, vx(1), vx(2), vx(3), ...
        or(1), or(2), or(3), radius);
    samples(j,:) = buff1(:);
    linIdx(j,:) = buff2(:);
end

end


function [orientations, samples, linIdx] = pickPatchParallelXZ(...
    img, voxels, patchWidth)
% Pick patches parallel to axis x and z

dims = size(img);
nSPerFile = size(voxels,1);
radius = floor(patchWidth / 2);

samples = zeros(nSPerFile, patchWidth^2);
linIdx = zeros(nSPerFile, patchWidth^2);
orientations = repmat([0 1 0], nSPerFile, 1);


for j = 1:nSPerFile

    vx = voxels(j,:);

    v1 = vx(1) - radius:vx(1) + radius;
    v1(v1<1) = 1; v1(v1>dims(1)) = dims(1); 
    v2 = vx(2);
    v3 = vx(3) - radius:vx(3) + radius;
    v3(v3<1) = 1; v3(v3>dims(3)) = dims(3); 

    [X,Y,Z] = meshgrid(v1, v2, v3);
    buff = sub2ind(size(img), X, Y, Z);
    linIdx(j,:) = buff(:);
    
    patch = img(v1,v2,v3);
    samples(j,:) = patch(:);
end

end

