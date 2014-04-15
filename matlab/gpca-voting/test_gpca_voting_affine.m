addpath helper_functions

clear;
close all;
clc;
noiseLevel = 0.06;
ambientDimension = 2;
isAffine = true;
aprioriSubspaceDimensions = [1, 1];

[X, aprioriSampleLabels, aprioriGroupBases] = generate_samples('ambientSpaceDimension', ambientDimension,...
    'groupSizes', 100*aprioriSubspaceDimensions,...
    'basisDimensions', aprioriSubspaceDimensions,...
    'noiseLevel', noiseLevel,...
    'isAffine', isAffine,...
    'minimumSubspaceAngle', pi/4);

homogenizedX = [X; ones(1,200)];
subspaceDimensions = [2,2];
[sampleLabels, groupBases] = gpca_voting(homogenizedX, subspaceDimensions,'postoptimization',true,'angleTolerance',0.4);


% Plot the gpca results
if ambientDimension<=3
    figureCount = figure();
    subplot(1,2,1);
    plot_data(X, aprioriSampleLabels);
    hold on;
    plot_subspaces(X, aprioriSampleLabels, aprioriSubspaceDimensions, true);
    hold off;
    title('Apriori Data');
    set(0,'CurrentFigure',figureCount); % Change back to the correct figure without changing its visibility.
    subplot(1,2,2);
    plot_data(homogenizedX, sampleLabels);
    hold on;
    plot_subspaces(homogenizedX, sampleLabels, subspaceDimensions, false);
    hold off;
    title('Final Segmentation')
end