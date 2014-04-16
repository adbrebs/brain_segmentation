function [mri, label] = openNII(fileName)

mriFolder = './mri/';
labelFolder = './label/';

disp(fileName)

mri = load_nii([mriFolder fileName]); 
label = load_nii([labelFolder fileName]);

end

