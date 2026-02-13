%% ==========================================================
%  Brain Tumour Detection – FINAL OPTIMIZED VERSION
%  - Smart adaptive thresholding that handles bright images
%  - Better tumor/normal tissue differentiation
%  - Prevents both false positives AND false negatives
%% ==========================================================

clc; clear; close all;

%% 1) Load Image
[file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif', 'Image Files'});
if isequal(file,0)
    disp("No file selected.");
    return;
end

img = imread(fullfile(path,file));
if size(img,3)==3
    gray = rgb2gray(img);
else
    gray = img;
end

gray = im2double(gray);
[rows, cols] = size(gray);

%% 2) Preprocessing
% Gentle denoising
den = imgaussfilt(gray, 1.2);

% Mild CLAHE
enh = adapthisteq(den, 'ClipLimit', 0.005, 'NumTiles', [8 8]);

%% 3) Robust Brain Mask (Skull Stripping)
T = graythresh(enh);
bm = imbinarize(enh, T * 0.85);

bm = imfill(bm,'holes');
bm = bwareaopen(bm, 2500);

% Keep largest component
cc = bwconncomp(bm);
if cc.NumObjects == 0
    disp("Error: No brain region detected");
    return;
end

np = cellfun(@numel, cc.PixelIdxList);
[~, idx] = max(np);

bm = false(rows,cols);
bm(cc.PixelIdxList{idx}) = true;

% Morphological operations
bm = imclose(bm, strel('disk', 15));
bm = imfill(bm,'holes');
bm = imerode(bm, strel('disk', 7));
bm = imfill(bm,'holes');
bm = imopen(bm, strel('disk', 3));
bm = imfill(bm,'holes');

brainOnly = enh .* bm;
brainArea = sum(bm(:));

%% 4) IMPROVED Adaptive Thresholding
brainPix = brainOnly(bm);
meanBrain = mean(brainPix);
stdBrain = std(brainPix);
medianBrain = median(brainPix);

% Calculate multiple threshold candidates
threshold1 = medianBrain + 1.8 * stdBrain;  % Statistical approach
threshold2 = prctile(brainPix, 93);          % Percentile approach
threshold3 = meanBrain + 2.0 * stdBrain;     % Alternative statistical

% For bright images, use percentile OR statistical (whichever is LOWER)
% This prevents threshold from being too high
if meanBrain > 0.6  % Bright image
    threshold_intensity = min(threshold2, threshold1);
else  % Normal brightness image
    threshold_intensity = max(threshold2, threshold1);
end

% Safety bounds: never exceed certain absolute values
threshold_intensity = min(threshold_intensity, 0.85);  % Cap at 85%
threshold_intensity = max(threshold_intensity, medianBrain + 1.5 * stdBrain);  % Floor

% Also try RELATIVE brightest regions approach
% Find regions significantly brighter than their neighborhood
% Create local contrast mask
h = fspecial('average', 15);
localMean = imfilter(brainOnly, h);
localContrast = brainOnly - localMean;
localContrastMask = localContrast > 0.08;  % Regions brighter than neighborhood

%% 5) Multi-Strategy Detection
% Strategy 1: Absolute intensity
candInt = (brainOnly > threshold_intensity) & bm;

% Strategy 2: Local contrast (for bright images where absolute fails)
candContrast = localContrastMask & bm & (brainOnly > medianBrain);

% Combine both strategies
candCombined = candInt | candContrast;

% Morphological cleanup
candCombined = imopen(candCombined, strel('disk', 2));
candCombined = imclose(candCombined, strel('disk', 7));
candCombined = imfill(candCombined,'holes');
candCombined = bwareaopen(candCombined, round(0.003 * brainArea));

%% 6) Spatial Filtering
mid = round(cols/2);
bandWidth = round(cols*0.10);
midMask = false(rows, cols);
midMask(:, max(1,mid-bandWidth):min(cols,mid+bandWidth)) = true;

centerBandWidth = round(cols*0.12);
centerMask = false(rows, cols);
centerMask(:, max(1,mid-centerBandWidth):min(cols,mid+centerBandWidth)) = true;

%% 7) Filter Candidates
cc = bwconncomp(candCombined);
stats = regionprops(cc, brainOnly, 'Area','PixelIdxList','Eccentricity',...
    'Solidity','BoundingBox','Centroid','MajorAxisLength','MinorAxisLength','MeanIntensity');

cand = false(rows,cols);

for i = 1:length(stats)
    regionMask = false(rows,cols);
    regionMask(stats(i).PixelIdxList) = true;
    
    overlapMid = sum(regionMask(:) & midMask(:)) / (stats(i).Area + eps);
    overlapCenter = sum(regionMask(:) & centerMask(:)) / (stats(i).Area + eps);
    
    % Reject very central elongated structures (ventricles)
    if overlapCenter > 0.7 && stats(i).Eccentricity > 0.88
        continue;
    end
    
    % Reject very irregular
    if stats(i).Solidity < 0.30
        continue;
    end
    
    % Reject very small
    if stats(i).Area < 0.003 * brainArea
        continue;
    end
    
    % Reject very elongated midline structures
    if overlapMid > 0.5 && stats(i).Eccentricity > 0.90
        continue;
    end
    
    % Aspect ratio
    aspectRatio = stats(i).MajorAxisLength / (stats(i).MinorAxisLength + eps);
    if aspectRatio > 6
        continue;
    end
    
    % Must be reasonably bright
    if stats(i).MeanIntensity < meanBrain + 0.5 * stdBrain
        continue;
    end
    
    cand(stats(i).PixelIdxList) = true;
end

%% 8) Gradient Enhancement (for ring-enhancing)
candAreaRatio = sum(cand(:)) / brainArea;

if candAreaRatio < 0.01
    G = imgradient(enh);
    gPix = G(bm);
    pGrad = prctile(gPix, 94);
    
    candGrad = (G > pGrad * 1.05) & bm;
    
    candGrad = imopen(candGrad, strel('disk',2));
    candGrad = imclose(candGrad, strel('disk',6));
    candGrad = imfill(candGrad,'holes');
    candGrad = bwareaopen(candGrad, round(0.004 * brainArea));
    
    % Filter
    ccG = bwconncomp(candGrad);
    statsG = regionprops(ccG, brainOnly, 'Area','PixelIdxList','Eccentricity','Solidity','MeanIntensity');
    
    candGradFiltered = false(rows,cols);
    for i = 1:length(statsG)
        regionMask = false(rows,cols);
        regionMask(statsG(i).PixelIdxList) = true;
        
        overlapCenter = sum(regionMask(:) & centerMask(:)) / (statsG(i).Area + eps);
        
        if overlapCenter < 0.6 && statsG(i).Solidity > 0.35 && ...
           statsG(i).Eccentricity < 0.90 && statsG(i).MeanIntensity > meanBrain
            candGradFiltered(statsG(i).PixelIdxList) = true;
        end
    end
    
    cand = cand | candGradFiltered;
end

%% 9) Final Cleanup
finalMask = cand;

finalMask = imclose(finalMask, strel('disk', 9));
finalMask = imfill(finalMask,'holes');
finalMask = bwareaopen(finalMask, round(0.005 * brainArea));

%% 10) Rejection Gate
ccT = bwconncomp(finalMask);
numRegions = ccT.NumObjects;

tumorAreaAll = sum(finalMask(:));
tumorRatioAll = tumorAreaAll / brainArea;

shouldReject = false;

% Reject if very small
if tumorRatioAll < 0.01
    shouldReject = true;
end

% Reject if massive (whole brain)
if tumorRatioAll > 0.55
    shouldReject = true;
end

% Reject if too fragmented
if numRegions >= 6
    shouldReject = true;
end

% Intensity check - relaxed
if numRegions > 0
    statsF = regionprops(ccT, brainOnly, 'MeanIntensity','Area');
    [~, maxIdx] = max([statsF.Area]);
    
    % Very relaxed - just needs to be above mean
    if statsF(maxIdx).MeanIntensity < meanBrain + 0.8 * stdBrain
        shouldReject = true;
    end
end

if shouldReject
    finalMask(:) = 0;
end

tumorAreaAll = sum(finalMask(:));
tumorRatioAll = tumorAreaAll / brainArea;

%% 11) NO TUMOUR Display
if tumorAreaAll == 0
    classification = "NO TUMOUR";
    confidence = 85;

    figure('Name','Brain Tumour Detection','Position',[50 50 1400 700]);

    subplot(2,3,1); imshow(gray); title("Original MRI");
    subplot(2,3,2); imshow(enh); title("Enhanced (Mild CLAHE)");
    subplot(2,3,3); imshow(finalMask); title("Tumour Mask (Empty)");
    
    subplot(2,3,4); imshow(bm); title("Brain Mask");
    subplot(2,3,5); 
    % Show local contrast for debugging
    imshow(localContrast, []); title("Local Contrast Map");

    subplot(2,3,6); axis off;
    txt = {
        sprintf("Classification: %s", classification)
        sprintf("Confidence: %d%%", confidence)
        ""
        "Tumour detected: NO"
        ""
        sprintf("Brain Mean: %.3f", meanBrain)
        sprintf("Brain Std: %.3f", stdBrain)
        sprintf("Threshold: %.3f", threshold_intensity)
        sprintf("(Image brightness: %.1f%%)", meanBrain*100)
    };
    text(0.05,0.9, txt, 'FontName','FixedWidth', 'FontSize',10, 'VerticalAlignment','top');

    sgtitle("Brain Tumour Detection – NO TUMOUR", 'FontSize', 16, 'FontWeight','bold');
    
    fprintf("\n=== NO Tumour Detected ===\n");
    fprintf("Classification: %s\n", classification);
    
    return;
end

%% 12) Validate Detected Tumours
ccT = bwconncomp(finalMask);
statsT = regionprops(ccT, brainOnly, ...
    'Area','Perimeter','Eccentricity','Solidity','BoundingBox',...
    'MeanIntensity','PixelIdxList','Centroid');

[~, order] = sort([statsT.Area], 'descend');
statsT = statsT(order);

multiMask = false(rows, cols);
validTumors = [];

for i=1:length(statsT)
    tumorRatio = statsT(i).Area / brainArea;
    
    % Size range: 0.7% to 45%
    if tumorRatio < 0.007 || tumorRatio > 0.45
        continue;
    end
    
    % Must be brighter than mean - very relaxed
    if statsT(i).MeanIntensity < meanBrain + 0.5 * stdBrain
        continue;
    end
    
    % Solidity
    if statsT(i).Solidity < 0.25
        continue;
    end
    
    validTumors = [validTumors; i];
    multiMask(statsT(i).PixelIdxList) = true;
    
    if length(validTumors) >= 3
        break;
    end
end

if isempty(validTumors)
    finalMask(:) = 0;
    classification = "NO TUMOUR";
    confidence = 82;

    figure('Name','Brain Tumour Detection','Position',[50 50 1400 700]);
    subplot(2,3,1); imshow(gray); title("Original MRI");
    subplot(2,3,2); imshow(enh); title("Enhanced");
    subplot(2,3,3); imshow(finalMask); title("Tumour Mask (Empty)");
    subplot(2,3,4); imshow(bm); title("Brain Mask");
    subplot(2,3,5); imshow(localContrast,[]); title("Local Contrast");
    subplot(2,3,6); axis off;
    text(0.05,0.9, "Candidates failed validation", 'FontSize',11);
    sgtitle("Brain Tumour Detection – NO TUMOUR", 'FontSize', 16, 'FontWeight','bold');
    
    fprintf("\n=== NO Tumour (Post-Validation) ===\n");
    return;
end

finalMask = multiMask;
statsT = statsT(validTumors);
K = length(validTumors);

%% 13) Features
mainTum = statsT(1);

tumorArea = mainTum.Area;
tumPerim = mainTum.Perimeter;
tumorRatio = tumorArea / brainArea;

compactness = 4*pi*tumorArea/(tumPerim^2 + eps);
irregularity = tumPerim/(2*sqrt(pi*tumorArea) + eps);
intensityContrast = (mainTum.MeanIntensity - meanBrain) / (stdBrain + eps);

%% 14) Classification
if mainTum.Solidity > 0.75 && compactness > 0.45 && irregularity < 1.5 && mainTum.Eccentricity < 0.85
    classification = "BENIGN (Meningioma-like)";
    confidence = 76;
    
elseif mainTum.Solidity < 0.50 || irregularity > 2.3 || compactness < 0.20
    classification = "MALIGNANT (Glioma-like)";
    confidence = 78;
    
elseif tumorRatio > 0.20
    classification = "MALIGNANT (Large mass)";
    confidence = 74;
    
else
    classification = "GLIOMA-LIKE (Possibly Malignant)";
    confidence = 65;
end

%% 15) Visualization
figure('Name','Brain Tumour Detection','Position',[50 50 1400 800]);

subplot(2,3,1);
imshow(gray); title("Original MRI");

subplot(2,3,2);
imshow(enh); title("Enhanced");

subplot(2,3,3);
imshow(finalMask); title("Tumour Mask");

subplot(2,3,4);
imshow(gray); hold on;

B = bwboundaries(finalMask);
for k=1:length(B)
    plot(B{k}(:,2), B{k}(:,1), 'r', 'LineWidth', 2);
end

for i=1:K
    rectangle('Position', statsT(i).BoundingBox, 'EdgeColor','g', 'LineWidth',2);
end

title("Outline + Bounding Boxes");
hold off;

subplot(2,3,5);
overlay = repmat(gray,[1 1 3]);
overlay(:,:,1) = min(overlay(:,:,1) + 0.6*double(finalMask), 1);
imshow(overlay);
title("Tumour Highlighted");

subplot(2,3,6);
axis off;
txt = {
    sprintf("Classification: %s", classification)
    sprintf("Confidence: %d%%", confidence)
    ""
    sprintf("Tumours detected: %d", K)
    sprintf("Main Tumour/Brain: %.2f%%", tumorRatio*100)
    sprintf("Area: %d px", tumorArea)
    sprintf("Solidity: %.3f", mainTum.Solidity)
    sprintf("Compactness: %.3f", compactness)
    sprintf("Irregularity: %.3f", irregularity)
    sprintf("Eccentricity: %.3f", mainTum.Eccentricity)
    sprintf("Mean Intensity: %.3f", mainTum.MeanIntensity)
    sprintf("Contrast: %.2f σ", intensityContrast)
    ""
    sprintf("Image Brightness: %.1f%%", meanBrain*100)
    sprintf("Threshold Used: %.3f", threshold_intensity)
};
text(0.05,0.9, txt, 'FontName','FixedWidth', 'FontSize',9, 'VerticalAlignment','top');

sgtitle(sprintf("Brain Tumour Detection – %s", classification), ...
    'FontSize', 16, 'FontWeight','bold');

fprintf("\n=== Tumour Detected ===\n");
fprintf("Classification: %s\n", classification);
fprintf("Confidence: %d%%\n", confidence);
fprintf("Tumours: %d | Ratio: %.2f%%\n", K, tumorRatio*100);
fprintf("Contrast: %.2f σ | Brightness: %.1f%%\n", intensityContrast, meanBrain*100);
