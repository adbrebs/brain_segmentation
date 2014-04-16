function [linIdx] = pickVoxelsRandomly(labelImg, nSPerFile)

inBrain = find(labelImg);
r = randi(length(inBrain), nSPerFile, 1);
linIdx = inBrain(r);

end

