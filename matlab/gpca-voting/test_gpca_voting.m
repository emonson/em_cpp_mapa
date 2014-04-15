addpath helper_functions

clear;
close all;
clc;

roundCount = 1;

seedTimeStamp = sum(100*clock); %% Generate a new random seed.
rand('state',seedTimeStamp);
randn('state', seedTimeStamp);


% Generate the data samples
ambientDimension = 3;
% Please put the subspace dimensions in descending order to get correct error estimation for this test
% This restriction is not required by the GPCA algorithm.
subspaceDimensions = [2 2 1];
groupCount = length(subspaceDimensions);
noiseLevel = 0.06;

basisErrorStat = 0;
segmentationErrorStat = 0;
for roundIndex=1:roundCount
    [X, aprioriSampleLabels, aprioriGroupBases] = generate_samples('ambientSpaceDimension', ambientDimension,...
        'groupSizes', 100*subspaceDimensions,...
        'basisDimensions', subspaceDimensions,...
        'noiseLevel', noiseLevel,...
        'avoidIntersection', true,...
        'minimumSubspaceAngle', pi/4);
    sampleCount = size(X,2);

    % Perform Ksubspaces on the data
    [sampleLabels, groupBases] = gpca_voting(X, subspaceDimensions,'postoptimization',true,'angleTolerance',0.4);

    % Reorder the bases and sample labels so that they are likely to agree with
    % the original ones.
    [bestMapping, errorProbability] = relabel_samples(aprioriSampleLabels, sampleLabels, subspaceDimensions);
    segmentationErrorStat = segmentationErrorStat + errorProbability;
    inverseMapping(bestMapping) = 1:length(bestMapping);
    
    fixedSampleLabels = -ones(1,sampleCount);
    inlierIndex=find(sampleLabels~=-1);
    fixedSampleLabels(inlierIndex) = bestMapping(sampleLabels(inlierIndex));
    for subspaceIndex=1:groupCount
        fixedBases{subspaceIndex} = groupBases{inverseMapping(subspaceIndex)};
    end
    
    % Calculate the underlying structure error of subspaces by the
    % difference of their space angles.
    basisErrorStat = basisErrorStat + average_basis_error(aprioriGroupBases, fixedBases);
    
    % Display up-to-date error statistics averaged over roundIndex.
    % The plot in 2- or 3-D below shows that a large portion of the
    % segmentation error is caused by the samples located at the
    % intersection, which by nature are ambiguous.
    disp(['Try #' num2str(roundIndex) '. Segmentation Error: ' num2str(segmentationErrorStat/roundIndex*100) '%.  Average Basis Error: ' num2str(basisErrorStat/roundIndex)]);
end

% Plot the gpca results
if ambientDimension<=3
    figureCount = figure();
    subplot(1,2,1);
    plot_data(X, aprioriSampleLabels);
    hold on;
    plot_subspaces(X, aprioriSampleLabels, subspaceDimensions, false);
    hold off;
    title('Apriori Data');
    set(0,'CurrentFigure',figureCount); % Change back to the correct figure without changing its visibility.
    subplot(1,2,2);
    plot_data(X, fixedSampleLabels);
    hold on;
    plot_subspaces(X, fixedSampleLabels, subspaceDimensions, false);
    hold off;
    title('Final Segmentation')
end

