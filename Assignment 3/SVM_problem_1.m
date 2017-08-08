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
