function pickVoxelFUN = pickVxFactory(mode)

switch mode
    case 'balanced'
        pickVoxelFUN = @pickVxBalanced;
    case 'inPlane'
        pickVoxelFUN = @pickVxInPlane;
    case 'random'
        pickVoxelFUN = @pickVxRandomly;
    otherwise
        disp('Error choice pickVoxelFUN')
end

end

% Prototype of pickVoxelFUN:
% inputs: 
%   - labelImg: 3D image of the region labels
%   - nClasses: number of different region labels
%   - nVPerFile: number of different voxels
%   - nPatchPerVoxel: number of patches per voxel
% outputs:
%   - voxels: 2D matrix. size: (nVPerFile*nPatchPerVoxel,3)
%       Each line represents a voxel. columns: x,y,z

function [voxels, vxLinIdx] = pickVxBalanced(labelImg, nClasses, nVPerFile, nPatchPerVoxel)

nPerRegion = nVPerFile / nClasses;

vxLinIdx = zeros(nPerRegion*nClasses,1);

for k = 1:nClasses
    region = find(labelImg == k);
    r = randi(length(region), nPerRegion, 1);
    vxLinIdx(1 + (k-1) * nPerRegion : k*nPerRegion) = region(r);
end

[voxels, vxLinIdx] = duplicateVx(vxLinIdx, labelImg, nPatchPerVoxel);

end

function [voxels, vxLinIdx] = pickVxInPlane(labelImg, nClasses, nVPerFile, nPatchPerVoxel)

y = 100;


end

function [voxels, vxLinIdx] = pickVxRandomly(labelImg, nClasses, nVPerFile, nPatchPerVoxel)

inBrain = find(labelImg);
r = randi(length(inBrain), nVPerFile, 1);
vxLinIdx = inBrain(r);

[voxels, vxLinIdx] = duplicateVx(vxLinIdx, labelImg, nPatchPerVoxel);

end





function [voxels, vxLinIdx] = duplicateVx(vxLinIdx, labelImg, nPatchPerVoxel)

vxLinIdx = vxLinIdx(ceil((1:nPatchPerVoxel*size(vxLinIdx,1))/nPatchPerVoxel), :);
[x, y, z] = ind2sub(size(labelImg), vxLinIdx);
voxels = [x, y, z];

end


