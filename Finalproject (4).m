% Final project (MRI image data)

%
%Close all open figures
close all
%Clear the workspace
clear
%Clear the command window
clc 
%% Loading Data

% Loading the MRI image dataset
% the data is organised into 4 gategories ('MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented')
imageFolder = '/MATLAB Drive/MRI image data/Alzheimer_s Dataset/train'; 
categories = {'MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'};
imds = imageDatastore(fullfile(imageFolder, categories), 'LabelSource', 'foldernames');
 

 %% Spliting Dataset

 % Split the dataset into training and test sets (Here, 70% of the data is used for training and 30% for testing)
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomize');


%% Preprocess Images

% Define the image size
imageSize = [227 227];

% Define the function for converting grayscale images to RGB
convertGrayscaleToRGB = @(img) cat(3, img, img, img);

% Create augmented datastores for training and testing
augmentedTrain = augmentedImageDatastore(imageSize, imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augmentedTest = augmentedImageDatastore(imageSize, imdsTest, 'ColorPreprocessing', 'gray2rgb');
%%  Loading Pre-trained CNN

% Load a pre-trained AlexNet model
net = alexnet;
inputSize = net.Layers(1).InputSize;

% Replace the final layers of the network
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories);
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];


%%  Specifying  Training Options

% Set the training options

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 1e-4, ... 
    'Verbose', false, ...
    'Plots', 'training-progress');

%% traning the network 

% Train the network
trainedNet = trainNetwork(augmentedTrain, layers, options);


%% Clasifying images

[YPred, scores] = classify(trainedNet, augmentedTest);


%% Calculate accuracy



YTest = imdsTest.Labels;
accuracy = mean(YPred == YTest);
disp(['Test accuracy: ', num2str(accuracy)]);



%  Display Sample Results
idx = randperm(numel(YTest), 9);
figure;
for i = 1:9
    subplot(3, 3, i);
    I = readimage(imdsTest, idx(i));
    imshow(I);
    label = YPred(idx(i));
    title(string(label));
end