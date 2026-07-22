%% Skin Cancer Detection with Attention Mechanism
% Multi-Class Classification: 7 skin lesion types with Spatial Attention
% MATLAB R2020b+
% Dataset: HAM10000

clear; clc; close all;

%% STEP 1: Paths & Parameters
dataPath = 'HAM10000';
trainRatio = 0.8;
imgSize = [224 224];
numEpochs = 30;
miniBatchSize = 32;
learnRate = 1e-3;
validationFrequency = 30;

%% STEP 2: Load Dataset
imds = imageDatastore(dataPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

disp('Dataset Summary:');
disp(countEachLabel(imds));

[imdsTrain, imdsVal] = splitEachLabel(imds, trainRatio, 'randomized');

disp('Training data:');
disp(countEachLabel(imdsTrain));
disp('Validation data:');
disp(countEachLabel(imdsVal));

%% STEP 3: Data Augmentation
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandRotation', [-20 20], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10]);

augimdsTrain = augmentedImageDatastore(imgSize, imdsTrain, ...
    'DataAugmentation', augmenter, ...
    'ColorPreprocessing', 'gray2rgb');

augimdsVal = augmentedImageDatastore(imgSize, imdsVal, ...
    'ColorPreprocessing', 'gray2rgb');

numClasses = numel(categories(imdsTrain.Labels));

%% STEP 4: Build ResNet-18 with Attention Mechanism
net = resnet18;
lgraph = layerGraph(net);

% Remove original final layers
lgraph = removeLayers(lgraph, ...
    {'fc1000','prob','ClassificationLayer_predictions'});

% ========== ADD ATTENTION MODULE ==========
attentionLayers = [
    globalAveragePooling2dLayer('Name','attention_gap')
    fullyConnectedLayer(32, 'Name', 'attention_fc1')
    reluLayer('Name', 'attention_relu')
    fullyConnectedLayer(512, 'Name', 'attention_fc2')
    sigmoidLayer('Name', 'attention_sigmoid')
    functionLayer(@(X) reshapeAttention(X), ...
        'Name', 'attention_reshape', ...
        'Formattable', true)
];

lgraph = addLayers(lgraph, attentionLayers);
lgraph = connectLayers(lgraph, 'pool5', 'attention_gap');

multiplyLayer = multiplicationLayer(2, 'Name', 'attention_multiply');
lgraph = addLayers(lgraph, multiplyLayer);

lgraph = connectLayers(lgraph, 'pool5', 'attention_multiply/in1');
lgraph = connectLayers(lgraph, 'attention_reshape', 'attention_multiply/in2');

% ========== FINAL CLASSIFICATION LAYERS ==========
finalLayers = [
    globalAveragePooling2dLayer('Name', 'final_gap')
    dropoutLayer(0.5, 'Name', 'dropout')
    fullyConnectedLayer(numClasses, ...
        'Name','fc_new', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
];

lgraph = addLayers(lgraph, finalLayers);
lgraph = connectLayers(lgraph, 'attention_multiply', 'final_gap');

%% STEP 5: Training Options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', learnRate, ...
    'Momentum', 0.9, ...
    'MaxEpochs', numEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'ValidationData', augimdsVal, ...
    'ValidationFrequency', validationFrequency, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10);

%% STEP 6: Train Network
disp('Training CNN with Attention Mechanism...');
tic;
trainedNet = trainNetwork(augimdsTrain, lgraph, options);
trainTime = toc;

save('trainedSkinCancerNet_Attention.mat', 'trainedNet');
fprintf('Training completed in %.2f minutes\n', trainTime/60);

%% STEP 7: Model Evaluation
[YPred, scores] = classify(trainedNet, augimdsVal);
YVal = imdsVal.Labels;

accuracy = mean(YPred == YVal);

classNames = categories(YVal);
fprintf('\n=== RESULTS WITH ATTENTION ===\n');
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
    
    if (precision + recall) > 0
        f1 = 2 * (precision * recall) / (precision + recall);
    else
        f1 = 0;
    end
    
    fprintf('%s - Recall: %.2f%%, Precision: %.2f%%, F1: %.2f%%\n', ...
        className, recall*100, precision*100, f1*100);
end

figure;
confusionchart(YVal, YPred);
title(sprintf('Confusion Matrix with Attention - Accuracy: %.2f%%', accuracy*100));

%% STEP 8: Visualize Attention (FIXED)
testIdx = 1;
testImgPath = imdsVal.Files{testIdx};
testImg = imread(testImgPath);
testImg = imresize(testImg, imgSize);

if size(testImg,3) == 1
    testImg = cat(3, testImg, testImg, testImg);
end

% Extract attention weights and squeeze
attentionWeights = activations(trainedNet, testImg, 'attention_sigmoid');
attentionWeights = squeeze(attentionWeights);  % FIX: Remove singleton dimensions

[predLabel, score] = classify(trainedNet, testImg);
confidence = max(score) * 100;
actualLabel = imdsVal.Labels(testIdx);

% Visualization
figure('Position', [100 100 1200 400]);

subplot(1,3,1);
imshow(testImg);
title('Original Image');

subplot(1,3,2);
bar(attentionWeights);  % Now works - attentionWeights is 1D
title('Channel Attention Weights');
xlabel('Channel Index');
ylabel('Attention Weight');
grid on;
ylim([0 1]);

subplot(1,3,3);
[sortedWeights, sortedIdx] = sort(attentionWeights, 'descend');
bar(sortedWeights(1:20));
title('Top 20 Important Channels');
xlabel('Rank');
ylabel('Weight');
xticklabels(sortedIdx(1:20));
xtickangle(45);
grid on;

sgtitle(sprintf('Actual: %s | Predicted: %s | Confidence: %.2f%%', ...
    char(actualLabel), char(predLabel), confidence));

%% END
fprintf('\n=== PERFORMANCE ===\n');
fprintf('Accuracy with Attention: %.2f%%\n', accuracy*100);
disp('Complete!');

%% Helper Function
function Y = reshapeAttention(X)
    Y = X;
end