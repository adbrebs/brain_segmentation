function [ fileList ] = listMICCAI(dirName)

mriFolder = [dirName 'miccai/mri/'];
labelFolder = [dirName 'miccai/label/'];

files = dir([mriFolder '*.nii']);
nFiles = length(files);
fileList = cell(nFiles,1);

for i = 1:nFiles
    fileList{i} = {[mriFolder files(i).name], [labelFolder files(i).name]};
end

end

