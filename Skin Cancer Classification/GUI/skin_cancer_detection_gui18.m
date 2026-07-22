%% SKIN CANCER DETECTION GUI
% Graphical user interface for HAM10000 7-class classification
% Compatible with trainedSkinCancerNet.mat

function skin_cancer_detection_gui18()
    
    %% Check if model exists
    if ~isfile('trainedSkinCancerNet18.mat')
        msgbox('Model not found! Train the network first using the training script.', 'Error');
        return;
    end
    
    % Load trained model
    load('trainedSkinCancerNet18.mat', 'trainedNet');
    fprintf('Model loaded successfully.\n');
    
    %% Create main figure
    fig = figure('Name', 'Skin Cancer Classification System', ...
        'NumberTitle', 'off', ...
        'Position', [100 100 1200 750], ...
        'MenuBar', 'none', ...
        'ToolBar', 'none', ...
        'Color', [0.94 0.94 0.94], ...
        'Resize', 'off');
    
    %% Title Panel
    uicontrol('Style', 'text', 'Parent', fig, ...
        'Position', [20 690 1160 50], ...
        'String', 'Automated Skin Cancer Classification System', ...
        'FontSize', 18, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.2 0.4 0.7], ...
        'ForegroundColor', [1 1 1]);
    
    %% Control Buttons
    btn_load = uicontrol('Style', 'pushbutton', 'Parent', fig, ...
        'Position', [20 630 160 45], ...
        'String', 'Load Image', ...
        'FontSize', 13, ...
        'FontWeight', 'bold', ...
        'BackgroundColor', [0.3 0.7 0.3], ...
        'ForegroundColor', [1 1 1], ...
        'Callback', @loadImage);
    
    btn_clear = uicontrol('Style', 'pushbutton', 'Parent', fig, ...
        'Position', [200 630 160 45], ...
        'String', 'Clear All', ...
        'FontSize', 13, ...
        'FontWeight', 'bold', ...
        'BackgroundColor', [0.8 0.3 0.3], ...
        'ForegroundColor', [1 1 1], ...
        'Callback', @clearAll);
    
    btn_export = uicontrol('Style', 'pushbutton', 'Parent', fig, ...
        'Position', [380 630 160 45], ...
        'String', 'Export Results', ...
        'FontSize', 13, ...
        'FontWeight', 'bold', ...
        'Enable', 'off', ...
        'Callback', @exportResults);
    
    %% Image Display Axes
    ax_img = axes('Parent', fig, 'Position', [0.05 0.38 0.40 0.40]);
    title(ax_img, 'Input Image', 'FontSize', 14, 'FontWeight', 'bold');
    axis(ax_img, 'off');
    box(ax_img, 'on');
    
    %% Prediction Chart Axes
    ax_pred = axes('Parent', fig, 'Position', [0.55 0.38 0.40 0.40]);
    title(ax_pred, 'Class Confidence Scores', 'FontSize', 14, 'FontWeight', 'bold');
    
    %% Results Panel
    txt_results = uicontrol('Style', 'text', 'Parent', fig, ...
        'Position', [20 20 1160 270], ...
        'String', 'No image loaded. Click "Load Image" to begin classification.', ...
        'FontSize', 11, ...
        'HorizontalAlignment', 'left', ...
        'BackgroundColor', [1 1 1], ...
        'ForegroundColor', [0 0 0], ...
        'FontName', 'Courier New', ...
        'Max', 2, ...
        'Enable', 'inactive');
    
    % Store results for export
    currentResults = struct();
    
    %% Callback: Load and Classify Image
    function loadImage(~, ~)
        [filename, filepath] = uigetfile({'*.jpg;*.png;*.jpeg', 'Image Files'}, ...
            'Select a skin lesion image');
        
        if filename == 0
            return;
        end
        
        try
            % Load image
            testImg = imread(fullfile(filepath, filename));
            
            % Preprocess image (match training code)
            imgSize = [224 224];
            testImg = imresize(testImg, imgSize);
            
            % Convert grayscale to RGB if needed
            if size(testImg, 3) == 1
                testImg = cat(3, testImg, testImg, testImg);
            end
            
            % Display original image
            axes(ax_img);
            imshow(testImg);
            title(ax_img, sprintf('Input: %s', filename), ...
                'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');
            
            % Classify image
            [predLabel, scores] = classify(trainedNet, testImg);
            confidence = max(scores) * 100;
            classNames = categories(predLabel);
            
            % Plot confidence scores
            axes(ax_pred);
            barHandle = bar(scores * 100, 'FaceColor', [0.2 0.5 0.8]);
            set(gca, 'XTickLabel', classNames, 'XTickLabelRotation', 45);
            ylabel('Confidence (%)', 'FontSize', 12, 'FontWeight', 'bold');
            xlabel('Class', 'FontSize', 12, 'FontWeight', 'bold');
            ylim([0 100]);
            grid on;
            
            % Add confidence threshold line
            hold on;
            yline(50, 'r--', 'LineWidth', 2, 'Label', 'Threshold');
            hold off;
            
            title(ax_pred, sprintf('Predicted: %s (%.2f%%)', ...
                char(predLabel), confidence), ...
                'FontSize', 12, 'FontWeight', 'bold');
            
            % Determine risk level
            risk_level = getRiskLevel(char(predLabel), confidence);
            
            % Format detailed results
            resultText = sprintf(['╔════════════════════════════════════════════════════════════════╗\n', ...
                '║          SKIN LESION CLASSIFICATION RESULTS                    ║\n', ...
                '╚════════════════════════════════════════════════════════════════╝\n\n', ...
                'FILE: %s\n', ...
                'TIMESTAMP: %s\n\n', ...
                '─────────────────────────────────────────────────────────────────\n', ...
                'PREDICTED CLASS: %s\n', ...
                'CONFIDENCE: %.2f%%\n', ...
                'RISK LEVEL: %s\n', ...
                '─────────────────────────────────────────────────────────────────\n\n', ...
                'CLASS PROBABILITY DISTRIBUTION:\n'], ...
                filename, datestr(now, 'yyyy-mm-dd HH:MM:SS'), ...
                char(predLabel), confidence, risk_level);
            
            % Add all class probabilities
            for i = 1:numel(classNames)
                resultText = sprintf('%s  • %-25s: %6.2f%%\n', ...
                    resultText, classNames{i}, scores(i)*100);
            end
            
            % Add recommendations
            resultText = sprintf(['%s\n', ...
                '─────────────────────────────────────────────────────────────────\n', ...
                'CLINICAL RECOMMENDATION:\n', ...
                '%s\n\n', ...
                '⚠️  DISCLAIMER:\n', ...
                'This is an automated screening tool for educational purposes.\n', ...
                'Always consult a qualified dermatologist for accurate medical\n', ...
                'diagnosis, treatment recommendations, and clinical decisions.\n', ...
                '─────────────────────────────────────────────────────────────────\n'], ...
                resultText, getRecommendation(char(predLabel), confidence));
            
            % Update results display
            set(txt_results, 'String', resultText);
            
            % Enable export button
            set(btn_export, 'Enable', 'on');
            
            % Store results for export
            currentResults.filename = filename;
            currentResults.filepath = filepath;
            currentResults.prediction = char(predLabel);
            currentResults.confidence = confidence;
            currentResults.scores = scores;
            currentResults.classNames = classNames;
            currentResults.timestamp = datestr(now);
            
        catch ME
            msgbox(sprintf('Error processing image: %s', ME.message), 'Error');
        end
    end
    
    %% Callback: Clear All
    function clearAll(~, ~)
        cla(ax_img);
        cla(ax_pred);
        axis(ax_img, 'off');
        title(ax_img, 'Input Image', 'FontSize', 14, 'FontWeight', 'bold');
        title(ax_pred, 'Class Confidence Scores', 'FontSize', 14, 'FontWeight', 'bold');
        set(txt_results, 'String', 'No image loaded. Click "Load Image" to begin classification.');
        set(btn_export, 'Enable', 'off');
        currentResults = struct();
    end
    
    %% Callback: Export Results
    function exportResults(~, ~)
        if isempty(fieldnames(currentResults))
            msgbox('No results to export.', 'Warning');
            return;
        end
        
        [file, path] = uiputfile('*.txt', 'Save Results As');
        if file == 0
            return;
        end
        
        fid = fopen(fullfile(path, file), 'w');
        fprintf(fid, 'SKIN CANCER DETECTION RESULTS\n');
        fprintf(fid, '==============================\n\n');
        fprintf(fid, 'File: %s\n', currentResults.filename);
        fprintf(fid, 'Timestamp: %s\n\n', currentResults.timestamp);
        fprintf(fid, 'Predicted Class: %s\n', currentResults.prediction);
        fprintf(fid, 'Confidence: %.2f%%\n\n', currentResults.confidence);
        fprintf(fid, 'All Class Probabilities:\n');
        for i = 1:numel(currentResults.classNames)
            fprintf(fid, '  %s: %.2f%%\n', currentResults.classNames{i}, ...
                currentResults.scores(i)*100);
        end
        fclose(fid);
        
        msgbox('Results exported successfully!', 'Success');
    end
    
    %% Helper: Get Risk Level
    function risk = getRiskLevel(predClass, conf)
        if contains(lower(predClass), 'mel') && conf > 70
            risk = '🔴 HIGH RISK';
        elseif conf >= 80
            risk = '🟡 MODERATE CONFIDENCE';
        elseif conf >= 60
            risk = '🟡 MODERATE CONFIDENCE';
        else
            risk = '⚪ LOW CONFIDENCE';
        end
    end
    
    %% Helper: Get Recommendation
    function rec = getRecommendation(predClass, conf)
        if contains(lower(predClass), 'mel')
            rec = ['URGENT: Potential melanoma detected. Immediate dermatologist\n', ...
                   'consultation recommended for biopsy and treatment planning.'];
        elseif conf >= 80
            rec = ['High confidence prediction. Recommend follow-up with\n', ...
                   'dermatologist for confirmation and monitoring.'];
        else
            rec = ['Low confidence prediction. Additional imaging or clinical\n', ...
                   'examination strongly recommended.'];
        end
    end
    
end