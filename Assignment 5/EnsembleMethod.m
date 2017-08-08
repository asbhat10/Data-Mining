%-------------------------------------------------------------------------%
clear variables;
close all;
clc;
%-------------------------------------------------------------------------%

%Load Data
path = 'E:\ASU\Courses\2Data Mining\Assignments\Assignment5\Data for Assignment 5 (Mini Project 3)\Handwritten Digits\';

trainset = load(strcat(path,'X_train.mat'));
trainset = trainset.X_train;
trainClass = load(strcat(path,'y_train.mat'));
trainClass = trainClass.y_train;

testset = load(strcat(path,'X_test.mat'));
testset = testset.X_test;
testClass = load(strcat(path,'y_test.mat'));
testClass = testClass.y_test;

classes = unique(trainClass);
numClasses = max(classes);

[IDX,D] = knnsearch(trainset,testset,'K',7,'Distance','euclidean' );

FinalClass = zeros(1,size(testset,1));
for c = 1:size(testset,1)
    count = zeros(numClasses,1);
    for i = 1:7
        count(trainClass(IDX(c,i))) = count(trainClass(IDX(c,i)))+1;
    end
    [M ,I] = max(count);   
    FinalClass(c) = I;
     
end

FinalClass_knn = FinalClass';
correctClass = (FinalClass_knn==testClass);
[totalRecords, ~] = size(testClass);
accuracy = sum(correctClass(:) == 1) * 100/totalRecords;

fprintf('Accuracy of KNN Algorithm : %f ', accuracy);


SVMModels = cell(numClasses,1);


for j = 1:numClasses;
    indx = (trainClass == classes(j)); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(trainset,indx,'ClassNames',[false true],'KernelFunction','Polynomial','PolynomialOrder',2);
end


N = size(testset,1);
ScoreMatrix = zeros(N,numClasses);

for j=1:numClasses
    [~,score] = predict(SVMModels{j},testset);
    ScoreMatrix(:,j) = score(:,2);
end;
N = size(testset,1);
ScoreMatrix = zeros(N,numClasses);

for j=1:numClasses
[~,score] = predict(SVMModels{j},testset);
ScoreMatrix(:,j) = score(:,2);
end;

[label,FinalClass_svm] = max(ScoreMatrix,[],2);


correctClass = (FinalClass_svm==testClass);

[totalRecords, ~] = size(testClass);
accuracy = sum(correctClass(:) == 1) * 100/totalRecords;

fprintf('\nAccuracy of SVM : %f ', accuracy);

target = zeros(size(classes,1),size(trainClass,1));

for k = 1:length(trainClass)  %row vector
   target(trainClass(k),k)=1;
end

net = feedforwardnet(25);
net = train(net,trainset',target);

testTargetClass=net(testset.');
FinalClass_ann = vec2ind(testTargetClass); 
FinalClass_ann = FinalClass_ann.';
correctClass = (FinalClass_ann==testClass);

[totalRecords, ~] = size(testClass);
accuracy = sum(correctClass(:) == 1) * 100/totalRecords;
fprintf('\nAccuracy of ANN : %f',accuracy);

EnsembleClass = zeros(size(testClass));
for c = 1:size(testset,1)
    count = zeros(numClasses,1);
    count(FinalClass_ann(c)) = count(FinalClass_ann(c))+1;
    count(FinalClass_knn(c)) = count(FinalClass_knn(c))+1;
    count(FinalClass_svm(c)) = count(FinalClass_svm(c))+1;
    [M ,I] = max(count);
        
    EnsembleClass(c) = I;
end

correctClass = (EnsembleClass==testClass);
accuracy = sum(correctClass(:) == 1) * 100/totalRecords;
fprintf('\nAccuracy of Ensemble method : %f',accuracy);

