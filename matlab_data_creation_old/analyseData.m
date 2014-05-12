%% Read the data
fileName = './../data/mridata.h5';
samples = h5read(fileName, '/inputs');

% Display one patch
%slice = reshape(samples(4,:), patchWidth, patchWidth);

mri = '../data/miccai/mri/1019.nii';
mask = '../data/miccai/label/1019.nii';

[mri, label] = openNII({mri,mask}, false);
[mri, label] = cropImg(mri, label);
% option.setvalue.idx = find(label.img);
% option.setvalue.val = label.img(option.setvalue.idx);
% option.useinterp = 1;
view_nii(mri)


figure
slice = extractSlice(mri.img,100,100,100,0,1,0,100);
colormap(gray)
imagesc(slice) 




patch = reshape(samples(3,:), patchWidth, patchWidth);
colormap(gray)
imagesc(patch)