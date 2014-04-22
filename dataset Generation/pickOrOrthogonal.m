function [ or ] = pickOrOrthogonal(nVPerFile, nOrPerVoxel)
% this function only works for nOrPerVoxel = 1 or 3

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

