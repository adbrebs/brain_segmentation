function [mri, label] = cropImg(mri, label)

lims = checkLimits(label.img);
lim1 = lims(1,1):lims(1,2);
lim2 = lims(2,1):lims(2,2);
lim3 = lims(3,1):lims(3,2);

mri.img = mri.img(lim1, lim2, lim3);
label.img = label.img(lim1, lim2, lim3);

end

