%-------------------------------------------------------------------------%
clear variables;
close all;
clc;
%-------------------------------------------------------------------------%
TrainData = load('E:\ASU\Courses\2Data Mining\Assignments\Assignment3\VidTIMIT\X_train.mat');
TrainData = TrainData.X_train;

TestData = load('E:\ASU\Courses\2Data Mining\Assignments\Assignment3\VidTIMIT\X_test.mat');
TestData = TestData.X_test;

TrainClass = load('E:\ASU\Courses\2Data Mining\Assignments\Assignment3\VidTIMIT\y_train.mat');
TrainClass = TrainClass.y_train;
TrainClass = TrainClass';

TestClass = load('E:\ASU\Courses\2Data Mining\Assignments\Assignment3\VidTIMIT\y_test.mat');
TestClass = TestClass.y_test;
TestClass = TestClass';


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