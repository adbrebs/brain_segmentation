function [mri, label] = openNII(fileCouple, reCodeClasses)

mriPath = fileCouple{1};
labelPath = fileCouple{2};

disp(mriPath)

mri = load_nii(mriPath); 
label = load_nii(labelPath);

if ~reCodeClasses
    return
end

% Change the class labels
classes = unique(label.img);
newClasses = 0:length(classes);
disp(length(classes))
[a,b] = ismember(label.img, classes);
label.img(a) = newClasses(b(a));

end

