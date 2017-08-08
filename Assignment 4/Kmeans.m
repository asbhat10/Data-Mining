%-------------------------------------------------------------------------%
clear variables;
close all;
clc;
%-------------------------------------------------------------------------%
%Loading the dataset and initialising the values
path = 'E:\ASU\Courses\2Data Mining\Assignments\Assignment4\Data for Assignment 4 (Mini Project 2)\Clustering\seeds.txt';
dataset = load(path);
rowSize = size(dataset,1);
columnSize = size(dataset,2);

%Enter the value of K
kprompt = 'Enter the value of K ';
k = input(kprompt);

%Distance matrix to store the distance of vectors from the centroid
distMatrix = zeros(rowSize,k);
tempSSE = 0;
SSE = zeros(10,1);

%Repeating K means for 10 random initialization
for i = 1:10
    clear centroid;
    
    %selecting random centroids
    randomCentroidIndices = randperm(size(dataset,1));
    for j = 1 : k
        centroid(j,:) = dataset(randomCentroidIndices(j),:);
    end
    
    %Loop until convergence 
    %Calculate the distance of each vector from the centroid
    for c=1:100
        distMatrix = pdist2(dataset,centroid);
        %get the minimum from each row
        [minValue,index] = min(distMatrix,[],2);
        
        if abs(tempSSE - SSE(i)) <= 0.001 && SSE(i) > 0
            SSE(i) = tempSSE;
            break;
        else
            SSE(i) = tempSSE;
            tempSSE = 0;
        end
   
        %Calculate new centroids
        for count = 1 : k
            sameCentroidCluster = find(index == count);
            if sameCentroidCluster 
                centroid(count,:) = mean(dataset(find(index == count),:),1);
            end
        end
        clear distMatrix;
        
        %Calculate SSE
        tempSSE = 0;
        for kCount = 1 : k
            for rowCount = 1 : rowSize
                if index(rowCount,1) == kCount
                    tempSSE = tempSSE + sum(( dataset(rowCount,:) - centroid(kCount,:)) .^ 2);
                end
            end
        end       
    end   
end
%Calculate average SSE
averageSSE = mean(SSE);
answer = ['Average SSE = ',num2str(averageSSE)];
disp(answer);


