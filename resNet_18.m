%% Early Automated Detection System For Skin Cancer Diagnosis through CNN
% Multi-Class Classification: 7 skin lesion types
% MATLAB R2020b+
% Dataset: HAM10000

clear; clc; close all;

%% STEP 1: Paths & Parameters
dataPath = 'HAM10000';           % dataset root
trainRatio = 0.8;
imgSize = [224 224];
numEpochs = 25;
miniBatchSize = 32;
learnRate = 1e-4;
validationFrequency = 30;

%% STEP 2: Load Dataset
imds = imageDatastore(dataPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

disp('Dataset Summary:');
disp(countEachLabel(imds));

% Split dataset AFTER loading
[imdsTrain, imdsVal] = splitEachLabel(imds, trainRatio, 'randomized');

disp('Training data:');
disp(countEachLabel(imdsTrain));
disp('Validation data:');
disp(countEachLabel(imdsVal));

%% STEP 3: Data Augmentation
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandRotation', [-10 10], ...
    'RandXTranslation', [-5 5], ...
    'RandYTranslation', [-5 5]);

augimdsTrain = augmentedImageDatastore(imgSize, imdsTrain, ...
    'DataAugmentation', augmenter, ...
    'ColorPreprocessing', 'gray2rgb');

augimdsVal = augmentedImageDatastore(imgSize, imdsVal, ...
    'ColorPreprocessing', 'gray2rgb');

%% STEP 4: Load Pretrained ResNet-18 (Multi-class)
net = resnet18;
lgraph = layerGraph(net);

% Get number of classes from dataset
numClasses = numel(categories(imdsTrain.Labels));  % Will be 7

% New final layers for all classes
newLayers = [
    fullyConnectedLayer(numClasses, ...
        'Name','fc_new', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
];

% Remove pretrained final layers
lgraph = removeLayers(lgraph, ...
    {'fc1000','prob','ClassificationLayer_predictions'});

% Add new layers
lgraph = addLayers(lgraph, newLayers);

% Connect new layers
lgraph = connectLayers(lgraph, 'pool5', 'fc_new');

%% STEP 5: Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate', learnRate, ...
    'MaxEpochs', numEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'ValidationData', augimdsVal, ...
    'ValidationFrequency', validationFrequency, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% STEP 6: Train Network
disp('Training CNN...');
tic;
trainedNet = trainNetwork(augimdsTrain, lgraph, options);
trainTime = toc;

save('trainedSkinCancerNet.mat', 'trainedNet');
fprintf('Training completed in %.2f minutes\n', trainTime/60);

%% STEP 7: Model Evaluation
[YPred, scores] = classify(trainedNet, augimdsVal);
YVal = imdsVal.Labels;

accuracy = mean(YPred == YVal);

% Per-class metrics
classNames = categories(YVal);
fprintf('\n=== RESULTS ===\n');
fprintf('Overall Accuracy: %.2f %%\n\n', accuracy*100);

fprintf('Per-Class Metrics:\n');
for i = 1:numel(classNames)
    className = classNames{i};
    tp = sum(YPred == className & YVal == className);
    fp = sum(YPred == className & YVal ~= className);
    fn = sum(YPred ~= className & YVal == className);
    
    if (tp + fn) > 0
        recall = tp / (tp + fn);
    else
        recall = 0;
    end
    
    if (tp + fp) > 0
        precision = tp / (tp + fp);
    else
        precision = 0;
    end
    
    fprintf('%s - Recall: %.2f%%, Precision: %.2f%%\n', ...
        className, recall*100, precision*100);
end

% Confusion Matrix
figure;
confusionchart(YVal, YPred);
title('Confusion Matrix - 7 Classes');

%% STEP 8: Single Image Testing
% Test on first validation image
testIdx = 1;
testImgPath = imdsVal.Files{testIdx};
testImg = imread(testImgPath);
testImg = imresize(testImg, imgSize);

% Ensure RGB
if size(testImg,3) == 1
    testImg = cat(3, testImg, testImg, testImg);
end

[predLabel, score] = classify(trainedNet, testImg);
confidence = max(score) * 100;
actualLabel = imdsVal.Labels(testIdx);

figure;
imshow(testImg);
title(sprintf('Actual: %s | Predicted: %s | Confidence: %.2f%%', ...
    char(actualLabel), char(predLabel), confidence));

%% END
disp('Training and evaluation complete!');