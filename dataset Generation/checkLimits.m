function [ limits ] = checkLimits(image)

limits = zeros(3,2);
dims = size(image);

f1 = @(i) image(i,:,:);
f2 = @(i) image(:,i,:);
f3 = @(i) image(:,:,i);
f = {f1,f2,f3};

for j = 1:3
    limits(j,1) = checkLimitsOneSide(f{j}, 1:dims(j));
    limits(j,2) = checkLimitsOneSide(f{j}, dims(j):-1:1);
end

end

function limits2D = checkLimitsOneSide(f, iterations)

for i = iterations
    if any(any(f(i))) == 1
        limits2D = i;
        return;
    end
end

limits2D = iterations(end);

end

