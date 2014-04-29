function pickTargetFUN = pickTargetFactory(mode)

switch mode
    case 'centered'
        pickTargetFUN = @pickTargetCentered;
    case 'proportion'
        pickTargetFUN = @pickTargetProportion;
    otherwise
        disp('Error choice pickTargetFUN')
end

end

% Prototype of pickTargetFUN:
% inputs: 
%   - labelImg: 3D image of the region labels
%   - vxLinIdx: vx linear indices (vector)
%   - patchLinIdx: patches linear indices (matrix)
%   - nClasses: number of classes
% outputs:
%   - targets: 2D matrix. size: (nVPerFile*nPatchPerVoxel,nClasses)
%       Each line represents the target of the corresponding point

function targets = pickTargetCentered(labelImg, vxLinIdx, patchLinIdx, nClasses)

nSamples = size(vxLinIdx,1);
targets = zeros(nSamples, nClasses);

idx = sub2ind(size(targets), 1:nSamples, labelImg(vxLinIdx));
targets(idx) = 1;

end


function targets = pickTargetProportion(labelImg, vxLinIdx, patchLinIdx, nClasses)

nSamples = size(patchLinIdx,1);

targets = zeros(nSamples, nClasses);

for i = 1:nSamples
    ta = tabulate(labelImg(patchLinIdx(i,:)));
    if ta(1,1) == 0
        ta = ta(2:end,:);
        ta(:,3) = ta(:,3) / sum(ta(:,3));
    end
    targets(i, ta(:,1)) = ta(:,3);
end

targets = targets / 100;

end

