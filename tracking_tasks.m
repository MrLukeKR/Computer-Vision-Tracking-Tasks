clear; close all; clc
%% Task 1
%% Getting started with Matlab: 
%% 1. Matlab Tutorial: http://www.mathworks.co.uk/videos/getting-started-with-matlab-68985.html?s_tid=prodeg_tutorial_ML_rp
%% 2. The Matlab Image Processing toolbox provides the low level functions needed to implement computer vision solutions. Follow 
%% the links to examples and video demonstrations: http://www.mathworks.co.uk/products/image/
%% 3. https://web.stanford.edu/class/cme001/handouts/matlab.pdf

%% Setting you up for the tasks
%% Assumes you're in a directory containing the .pgm files.
%% Saves the images in a uint8 array called 'data'.
filenames = dir('*.pgm');
filenames = sort({filenames.name});

%% Read the first to see how large it should be
im = imread(filenames{1});
data = repmat(uint8(0),[size(im,1) size(im,2) length(filenames)]);
normalisedData = repmat(uint8(0),[size(im,1) size(im,2) length(filenames)]);
for ii = 1:length(filenames)
        data(:,:,ii) = imread(filenames{ii});
end

%% Task 2
%% Plot frame 1
frameOne = data(:,:,1);
imshow(frameOne);
hold on;

%implay(data(:,:,:));

%% Task 3
%% Draw a rectangle around the face on top of the image.
%% The rectangle should be located at location (row, col) = (76, 142)
%% and should have (width, height) = (32, 32) pixels
rectangle('Position',[142,76,32,32], 'EdgeColor', [1,0,0]);
hold off;

%% Task 4 
%% Extract the template face from the image and plot template
figure;
template = frameOne(76:107, 142:173,1);
imshow(template);

%% Task 5
%% Normalise image and template by dividing by 255
normalisedTemplate = zeros(size(template,1),size(template,2));
normalisedFrameOne = zeros(size(frameOne,1),size(frameOne,2));
normalisedData = zeros(size(data,1), size(data,2),size(data,3));

normalisedData(:,:,:) = data(:,:,:) / 255;
normalisedFrameOne(:,:) = frameOne(:,:) / 255;

normalisedTemplate(:,:) = template(:,:) / 255;

figure;
imshow(normalisedTemplate);

figure;
imshow(normalisedFrameOne);

%% Do Euclidean distance based-matching between the template and frame 1
%% Plot the matching surface as a 2D image and find the location that
%% corresponds to the min value
%% What do you observe? - It's computationally expensive and slow to perform so many Euclidean distance calculations

xDim = size(normalisedFrameOne,1) - size(normalisedTemplate,1);
yDim = size(normalisedFrameOne,2) - size(normalisedTemplate,2);

distances = zeros(xDim, yDim);

figure;


for x = 1 : xDim
   for y = 1 : yDim 
      subArray = normalisedFrameOne(x:x+31,y:y+31);
       distances(x,y) = sqrt(sum(sum((subArray-normalisedTemplate).^2)));
   end
end

[val, loc] = min(distances(:));
[locX,locY] = ind2sub(size(distances),loc);



imshow(frameOne);
rectangle('Position',[locY,locX,32,32], 'EdgeColor', [1,0,0]);
hold off;

%% Task 6
%% Now replace Euclidean Distance with cross-correlation based-matching between the template and frame 1
%% See https://en.wikipedia.org/wiki/Cross-correlation
%% Plot the correlation surface as a 2D image and find the location that
%% corresponds to the max value
%% What do you observe? - It doesn't work
xDim = size(normalisedFrameOne,1) - size(normalisedTemplate,1);
yDim = size(normalisedFrameOne,2) - size(normalisedTemplate,2);

correlations = zeros(xDim, yDim);

figure;

for x = 1 : xDim
   for y = 1 : yDim 
      subArray = normalisedFrameOne(x:x+31,y:y+31);
      correlations(x,y) = sum(sum(subArray.*normalisedTemplate));
   end
end
   
[val, loc] = max(correlations(:));
[locX,locY] = ind2sub(size(correlations),loc);

imshow(frameOne);
rectangle('Position',[locY,locX,32,32], 'EdgeColor', [1,0,0]);
hold off;


%% Task 7
%% Repeat the above but replace correlation with zero-normalised cross-correlation
%% See https://en.wikipedia.org/wiki/Cross-correlation

xDim = size(normalisedFrameOne,1) - size(normalisedTemplate,1);
yDim = size(normalisedFrameOne,2) - size(normalisedTemplate,2);

correlations = zeros(xDim, yDim);

figure;

for x = 1 : xDim
   for y = 1 : yDim 
      subArray = normalisedFrameOne(x:x+31,y:y+31);
            correlations(x,y) = (1/(32*32)) * sum(sum((1 / (std2(subArray) * std2(normalisedTemplate))) * ((subArray-mean(subArray)).*(normalisedTemplate-mean(normalisedTemplate)))));
   end
end
   
[val, loc] = max(correlations(:));
[locX,locY] = ind2sub(size(correlations),loc);

imshow(frameOne);
rectangle('Position',[locY,locX,32,32], 'EdgeColor', [1,0,0]);
hold off;

%% Task 8
%% Do tracking without template update
%% What do you observe?
trackedImages = uint8(zeros(size(data,1),size(data,2), 3,size(data,3)));

for z = 1 : size(normalisedData,3)
for x = 1 : xDim
   for y = 1 : yDim 
      subArray = normalisedData(x:x+31,y:y+31, z);
      correlations(x,y) = (1/(32*32)) * sum(sum((1 / (std2(subArray) * std2(normalisedTemplate))) * ((subArray-mean(subArray)).*(normalisedTemplate-mean(normalisedTemplate)))));
   end
end
    
[val, loc] = max(correlations(:));
[locX,locY] = ind2sub(size(correlations),loc);

trackedImages(:,:,:,z) = uint8(insertShape(data(:,:,z),'rectangle',[locY,locX,32,32],'LineWidth',1,'Color','red'));
end

implay(trackedImages);

%% Task 9
%% Do tracking with template update
%% What do you observe?
trackedImages = uint8(zeros(size(data,1),size(data,2), 3,size(data,3)));

for z = 1 : size(normalisedData,3)
for x = 1 : xDim
   for y = 1 : yDim 
      subArray = normalisedData(x:x+31,y:y+31, z);
      correlations(x,y) = (1/(32*32)) * sum(sum((1 / (std2(subArray) * std2(normalisedTemplate))) * ((subArray-mean(subArray)).*(normalisedTemplate-mean(normalisedTemplate)))));
   end
end
    
[val, loc] = max(correlations(:));
[locX,locY] = ind2sub(size(correlations),loc);

template = data(locX:locX+31, locY:locY+31, z);
normalisedTemplate(:,:) = template(:,:) / 255;

trackedImages(:,:,:,z) = uint8(insertShape(data(:,:,z),'rectangle',[locY,locX,32,32],'LineWidth',1,'Color','red'));
end

implay(trackedImages);

%% Task 10
%% Try to combine the two approaches. Think about how.

%% Task 11
%% Try other sequences from http://www.cs.toronto.edu/~dross/ivt/
%% Use the videos provided as a sequence of .pgm image files

%% Task 12
%% Try using HOG features
%% See help for extractHOGFeatures function
%% Consider narrowing down the search region to improve efficiency

%% Task 13
%% Try SIFT-based matching
%% SIFT is not supported by Matlab so maybe you want to use SURF features

%% Task 14
%% Use motion features (optical flow) to improve matching
%% For which sequences flow can be useful?
