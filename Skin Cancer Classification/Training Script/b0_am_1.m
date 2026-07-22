%% Optimized Training - NO Early Stopping
% Will train all 75 epochs guaranteed

clear; clc; close all;
rng(42);

%% Parameters
dataPath = 'HAM10000';
trainRatio = 0.85;
imgSize = [224 224];
numEpochs = 75;                 % Will train ALL 75 epochs
miniBatchSize = 32;
initialLearnRate = 1e-3;
validationFrequency = 30;

%% Load Dataset
imds = imageDatastore(dataPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

fprintf('========== DATASET ==========\n');
disp(countEachLabel(imds));

[imdsTrain, imdsVal] = splitEachLabel(imds, trainRatio, 'randomized');
fprintf('\nTrain: %d | Val: %d\n', length(imdsTrain.Files), length(imdsVal.Files));

%% Data Augmentation
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandRotation', [-20 20], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10], ...
    'RandXScale', [0.95 1.05], ...
    'RandYScale', [0.95 1.05]);

augimdsTrain = augmentedImageDatastore(imgSize, imdsTrain, ...
    'DataAugmentation', augmenter, ...
    'ColorPreprocessing', 'gray2rgb');

augimdsVal = augmentedImageDatastore(imgSize, imdsVal, ...
    'ColorPreprocessing', 'gray2rgb');

numClasses = numel(categories(imdsTrain.Labels));

%% Build Network
net = efficientnetb0;
lgraph = layerGraph(net);
layerNames = {lgraph.Layers.Name};

% Detect layers
poolLayerName = '';
possibleNames = {
    'efficientnet-b0|model|head|global_average_pooling2d|GlobAveragePool'
    'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool'
};

for i = 1:length(possibleNames)
    if any(strcmp(layerNames, possibleNames{i}))
        poolLayerName = possibleNames{i};
        break;
    end
end

if isempty(poolLayerName)
    poolIdx = find(contains(layerNames, 'pool', 'IgnoreCase', true) & ...
                   contains(layerNames, 'average', 'IgnoreCase', true));
    poolLayerName = layerNames{poolIdx(end)};
end

connections = lgraph.Connections;
poolInput = connections(strcmp(connections.Destination, poolLayerName), :);
featureLayerName = poolInput.Source{1};

fprintf('\n========== BUILDING NETWORK ==========\n');
fprintf('Feature layer: %s\n', featureLayerName);

% Remove old layers
featureIdx = find(strcmp(layerNames, featureLayerName));
layersToRemove = layerNames(featureIdx+1:end);
for i = 1:length(layersToRemove)
    try lgraph = removeLayers(lgraph, layersToRemove{i}); catch; end
end

numFeatures = 1280;

% Direct path
directPath = globalAveragePooling2dLayer('Name', 'gap_direct');
lgraph = addLayers(lgraph, directPath);
lgraph = connectLayers(lgraph, featureLayerName, 'gap_direct');

% Attention path
attentionPath = [
    globalAveragePooling2dLayer('Name', 'gap_attention')
    fullyConnectedLayer(80, 'Name', 'att_fc1', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    reluLayer('Name', 'att_relu')
    fullyConnectedLayer(numFeatures, 'Name', 'att_fc2', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    sigmoidLayer('Name', 'att_sigmoid')
];
lgraph = addLayers(lgraph, attentionPath);
lgraph = connectLayers(lgraph, featureLayerName, 'gap_attention');

% Multiply
multiplyLayer = multiplicationLayer(2, 'Name', 'att_multiply');
lgraph = addLayers(lgraph, multiplyLayer);
lgraph = connectLayers(lgraph, 'gap_direct', 'att_multiply/in1');
lgraph = connectLayers(lgraph, 'att_sigmoid', 'att_multiply/in2');

% Classification
classifier = [
    fullyConnectedLayer(512, 'Name', 'fc1', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.5, 'Name', 'drop1')
    fullyConnectedLayer(numClasses, 'Name', 'fc_out', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];
lgraph = addLayers(lgraph, classifier);
lgraph = connectLayers(lgraph, 'att_multiply', 'fc1');

fprintf('✓ Network built!\n');

%% Training Options - NO EARLY STOPPING
options = trainingOptions('sgdm', ...
    'InitialLearnRate', initialLearnRate, ...
    'Momentum', 0.9, ...
    'MaxEpochs', numEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'ValidationData', augimdsVal, ...
    'ValidationFrequency', validationFrequency, ...
    'Shuffle', 'every-epoch', ...
    'L2Regularization', 5e-4, ...
    'Verbose', true, ...
    'VerboseFrequency', 30, ...
    'Plots', 'training-progress', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 20, ...
    'ExecutionEnvironment', 'auto');
    % NO ValidationPatience - will train all 75 epochs!

%% Train
fprintf('\n========================================\n');
fprintf('TRAINING CONFIGURATION\n');
fprintf('========================================\n');
fprintf('Max Epochs: %d (NO early stopping)\n', numEpochs);
fprintf('Initial LR: %.4f\n', initialLearnRate);
fprintf('LR Schedule: Drop 50%% every 20 epochs\n');
fprintf('Batch Size: %d\n', miniBatchSize);
fprintf('========================================\n\n');

tic;
trainedNet = trainNetwork(augimdsTrain, lgraph, options);
trainTime = toc;

save('trainedEfficientNet_Full.mat', 'trainedNet');
fprintf('\n✓ Completed in %.2f minutes\n', trainTime/60);

%% Evaluate
[YPred, scores] = classify(trainedNet, augimdsVal);
YVal = imdsVal.Labels;
accuracy = mean(YPred == YVal);

fprintf('\n========================================\n');
fprintf('FINAL ACCURACY: %.2f%%\n', accuracy*100);
fprintf('========================================\n');

classNames = categories(YVal);
fprintf('\n%-20s %8s %8s %8s\n', 'Class', 'Recall', 'Prec.', 'F1');
fprintf('%s\n', repmat('-', 1, 55));

for i = 1:numClasses
    tp = sum(YPred == classNames{i} & YVal == classNames{i});
    fp = sum(YPred == classNames{i} & YVal ~= classNames{i});
    fn = sum(YPred ~= classNames{i} & YVal == classNames{i});
    
    recall = tp / max(tp + fn, 1);
    precision = tp / max(tp + fp, 1);
    f1 = 2 * (precision * recall) / max(precision + recall, eps);
    
    fprintf('%-20s %7.2f%% %7.2f%% %7.2f%%\n', ...
        char(classNames{i}), recall*100, precision*100, f1*100);
end

% Confusion Matrix
figure;
cm = confusionchart(YVal, YPred);
cm.Title = sprintf('Accuracy: %.2f%%', accuracy*100);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

fprintf('\n✓ Training complete - reached all %d epochs!\n', numEpochs);