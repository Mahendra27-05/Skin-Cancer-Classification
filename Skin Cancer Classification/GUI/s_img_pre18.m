%% SINGLE IMAGE PREDICTION
% Load trained model and classify a new image
% Compatible with HAM10000 7-class model
close all; clc;

%% STEP 1: Load the trained model
load('trainedSkinCancerNet18.mat', 'trainedNet');
fprintf('Model loaded successfully.\n');

%% STEP 2: Select image
[filename, filepath] = uigetfile({'*.jpg;*.png;*.jpeg', 'Image Files'}, ...
    'Select a skin lesion image');

if filename == 0
    fprintf('No image selected.\n');
    return;
end

imagePath = fullfile(filepath, filename);
testImg = imread(imagePath);

%% STEP 3: Preprocess image
imgSize = [224 224];
testImg = imresize(testImg, imgSize);

% Convert grayscale to RGB if needed
if size(testImg, 3) == 1
    testImg = cat(3, testImg, testImg, testImg);
end

%% STEP 4: Classify image
[predLabel, scores] = classify(trainedNet, testImg);
confidence = max(scores) * 100;

%% STEP 5: Display results
figure('Position', [100 100 1200 500]);

% Show input image
subplot(1, 2, 1);
imshow(testImg);
title('Input Image', 'FontSize', 14, 'FontWeight', 'bold');

% Show prediction scores
subplot(1, 2, 2);
classNames = categories(predLabel);
bar(scores * 100);
set(gca, 'XTickLabel', classNames, 'XTickLabelRotation', 45);
ylabel('Confidence (%)', 'FontSize', 12);
xlabel('Class', 'FontSize', 12);
title(sprintf('Prediction: %s\nConfidence: %.2f%%', ...
    char(predLabel), confidence), ...
    'FontSize', 14, 'FontWeight', 'bold');
grid on;

% Add confidence threshold line
hold on;
yline(50, 'r--', 'LineWidth', 1.5);
hold off;

%% STEP 6: Print detailed results
fprintf('\n========== PREDICTION RESULTS ==========\n');
fprintf('Image: %s\n', filename);
fprintf('Predicted Class: %s\n', char(predLabel));
fprintf('Confidence: %.2f%%\n\n', confidence);

fprintf('All Class Probabilities:\n');
for i = 1:numel(classNames)
    fprintf('  %s: %.2f%%\n', classNames{i}, scores(i)*100);
end
fprintf('=========================================\n\n');

% Risk assessment based on confidence
if confidence >= 80
    fprintf('HIGH CONFIDENCE prediction.\n');
elseif confidence >= 60
    fprintf('MODERATE CONFIDENCE prediction.\n');
else
    fprintf('LOW CONFIDENCE prediction. Consider additional evaluation.\n');
end

fprintf('\nNote: This is an automated screening tool.\n');
fprintf('Please consult a dermatologist for medical diagnosis.\n');