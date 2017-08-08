%-------------------------------------------------------------------------%
clear variables;
close all;
clc;
%-------------------------------------------------------------------------%
TrainData = dlmread('E:\ASU\Courses\2Data Mining\Assignments\Assignment3\Human Activity Recognition\X_train.txt');
TestData = dlmread('E:\ASU\Courses\2Data Mining\Assignments\Assignment3\Human Activity Recognition\X_test.txt');

TrainClass = dlmread('E:\ASU\Courses\2Data Mining\Assignments\Assignment3\Human Activity Recognition\y_train.txt');
TestClass = dlmread('E:\ASU\Courses\2Data Mining\Assignments\Assignment3\Human Activity Recognition\y_test.txt');

classes = unique(TrainClass);
maxNumClasses = max(classes);

[IDX,D] = knnsearch(TrainData,TestData,'K',5,'Distance','euclidean' );
errorCount = 0;
for c = 1:size(TestData,1)
    count = zeros(maxNumClasses,1);
    for i = 1:5
        count(TrainClass(IDX(c,i))) = count(TrainClass(IDX(c,i)))+1;
    end
    [M ,I] = max(count);
        
    FinalClass(c) = I;
    
    
end

FinalClass = FinalClass';
correctClass = (FinalClass==TestClass);
[totalRecords, ~] = size(TestClass);
accuracy = sum(correctClass(:) == 1) * 100/totalRecords;

fprintf('Accuracy of KNN Algorithm : %f ', accuracy);