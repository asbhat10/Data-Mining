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
SVMModels = cell(maxNumClasses,1);


for j = 1:maxNumClasses;
    indx = (TrainClass == classes(j)); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(TrainData,indx,'ClassNames',[false true],'KernelFunction','Polynomial','PolynomialOrder',2);
end


N = size(TestData,1);
ScoreMatrix = zeros(N,maxNumClasses);

for j=1:maxNumClasses
[~,score] = predict(SVMModels{j},TestData);
ScoreMatrix(:,j) = score(:,2);
end;

[label,maxScore] = max(ScoreMatrix,[],2);


correctClass = (maxScore==TestClass);

[totalRecords, ~] = size(TestClass);
accuracy = sum(correctClass(:) == 1) * 100/totalRecords;

fprintf('Accuracy of SVM : %f ', accuracy);
