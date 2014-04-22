%% Read the data
fileName = './../data/mridata.h5';
samples = h5read(fileName, '/inputs');

% Display one patch
%slice = reshape(samples(4,:), patchWidth, patchWidth);


[mri, label] = openNII('1000.nii');
[mri, label] = cropImg(mri, label);
view_nii(mri)

figure
slice = extractSlice(mri.img,100,100,100,0,1,0,100);
colormap(gray)
imagesc(slice) 