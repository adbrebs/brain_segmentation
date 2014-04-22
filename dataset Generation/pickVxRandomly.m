function [linIdx] = pickVxRandomly(labelImg, nClasses, nVPerFile)

inBrain = find(labelImg);
r = randi(length(inBrain), nVPerFile, 1);
linIdx = inBrain(r);

end

