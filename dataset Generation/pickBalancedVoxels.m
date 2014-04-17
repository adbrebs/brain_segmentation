function [ linIdx ] = pickBalancedVoxels(labelImg, nClasses, nVPerFile)

nPerRegion = nVPerFile / nClasses;

linIdx = zeros(nPerRegion*nClasses,1);

for k = 1:nClasses
    region = find(labelImg == k);
    r = randi(length(region), nPerRegion, 1);
    linIdx(1 + (k-1) * nPerRegion : k*nPerRegion) = region(r);
end

end

