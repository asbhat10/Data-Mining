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

target = zeros(size(classes,1),size(TrainClass,1));

for k = 1:length(TrainClass)  %row vector
   target(TrainClass(k),k)=1;
end

net = feedforwardnet(25);
net = train(net,TrainData',target);

testTargetClass=net(TestData.');
predictedClass = vec2ind(testTargetClass); 
predictedClass = predictedClass.';
correctClass = (predictedClass==TestClass);

[totalRecords, ~] = size(TestClass);
accuracy = sum(correctClass(:) == 1) * 100/totalRecords;
fprintf('Accuracy of ANN : %f',accuracy);
