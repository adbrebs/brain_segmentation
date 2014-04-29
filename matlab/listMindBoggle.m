function [fileList] = listMindBoggle(dirName)

fileList = {};

dirData = dir(dirName);
dirIndex = [dirData.isdir];
potFiles = {dirData(~dirIndex).name}';

if ~isempty(potFiles)
    label = dir([dirName '/labels.DKT31.manual.MNI152.nii']);
    if ~isempty(label)
        mri = dir([dirName '/t1weighted_brain.MNI152.nii']);
        fileList = {{fullfile(dirName,mri.name), fullfile(dirName,label.name)}};
    end
end

subDirs = {dirData(dirIndex).name};
validIndex = ~ismember(subDirs,{'.','..'});

for iDir = find(validIndex)
    nextDir = fullfile(dirName,subDirs{iDir});
    fileList = [fileList; listMindBoggle(nextDir)];
end

end

