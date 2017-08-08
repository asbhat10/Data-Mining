%-------------------------------------------------------------------------%
clear variables;
close all;
clc;
%-------------------------------------------------------------------------%

%Load Data
path = 'E:\ASU\Courses\2Data Mining\Assignments\Assignment5\Data for Assignment 5 (Mini Project 3)\Multi Label Scene Data\'
trainset = load(strcat(path,'X_train.mat'));
trainset = trainset.X_train;
trainClass = load(strcat(path,'y_train.mat'));
trainClass = trainClass.y_train;

testset = load(strcat(path,'X_test.mat'));
testset = testset.X_test;
testClass = load(strcat(path,'y_test.mat'));
testClass = testClass.y_test;

noOfClasses = 6;

SVMModels = cell(noOfClasses,1);


for j = 1:noOfClasses;
    SVMModels{j} = fitcsvm(trainset,trainClass(:,j),'ClassNames',[false true],'KernelFunction','Polynomial','PolynomialOrder',2);
end


N = size(testset,1);
ScoreMatrix = zeros(N,noOfClasses);

for j=1:noOfClasses
    [~,score] = predict(SVMModels{j},testset);
    ScoreMatrix(:,j) = score(:,2);
end;

predictedLabel = zeros(size(ScoreMatrix,1),size(ScoreMatrix,2));

for i = 1:size(ScoreMatrix,1)
    for k = 1:size(ScoreMatrix,2)
        if ScoreMatrix(i,k) < 0
            predictedLabel(i,k) = 0;
        else
            predictedLabel(i,k) = 1;
        end
    end
end


accuracy = 0;
for i = 1:size(predictedLabel,1)
    intersection = 0;
    union = 0;
    for k = 1:size(predictedLabel,2)
        if predictedLabel(i,k) == 1 && testClass(i,k) == 1
            intersection = intersection + 1;
        end
        if predictedLabel(i,k) ~= 0 || testClass(i,k) ~= 0
            union = union + 1;
        end
    end
    accuracy = accuracy + (intersection/union);
end

totalRecords = size(predictedLabel,1);
accuracy = (accuracy*100)/totalRecords;
fprintf('Accuracy of SVM Polynomial kernel: %f ', accuracy);


SVMModels = cell(noOfClasses,1);


for j = 1:noOfClasses;
    SVMModels{j} = fitcsvm(trainset,trainClass(:,j),'ClassNames',[false true],'KernelFunction','gaussian','KernelScale','auto');
end


N = size(testset,1);
ScoreMatrix = zeros(N,noOfClasses);

for j=1:noOfClasses
    [~,score] = predict(SVMModels{j},testset);
    ScoreMatrix(:,j) = score(:,2);
end;

predictedLabel = zeros(size(ScoreMatrix,1),size(ScoreMatrix,2));

for i = 1:size(ScoreMatrix,1)
    for k = 1:size(ScoreMatrix,2)
        if ScoreMatrix(i,k) < 0
            predictedLabel(i,k) = 0;
        else
            predictedLabel(i,k) = 1;
        end
    end
end


accuracy = 0;
for i = 1:size(predictedLabel,1)
    intersection = 0;
    union = 0;
    for k = 1:size(predictedLabel,2)
        if predictedLabel(i,k) == 1 && testClass(i,k) == 1
            intersection = intersection + 1;
        end
        if predictedLabel(i,k) ~= 0 || testClass(i,k) ~= 0
            union = union + 1;
        end
    end
    accuracy = accuracy + (intersection/union);
end

totalRecords = size(predictedLabel,1);
accuracy = (accuracy*100)/totalRecords;
fprintf('\n Accuracy of SVM Gaussian kernel: %f ', accuracy);

