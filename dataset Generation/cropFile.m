function [mri, label, mask] = cropFile(fileName)

mriFolder = './mri/';
maskFolder = './mask/';
labelFolder = './label/';

disp(fileName)
mri = load_nii([mriFolder fileName]); 
mask = load_nii([maskFolder fileName]);
label = load_nii([labelFolder fileName]);

lims = checkLimits(mask.img);
lim1 = lims(1,1):lims(1,2);
lim2 = lims(2,1):lims(2,2);
lim3 = lims(3,1):lims(3,2);

mri.img = mri.img(lim1, lim2, lim3);
label.img = label.img(lim1, lim2, lim3);
mask.img = logical(mask.img(lim1, lim2, lim3));

end

