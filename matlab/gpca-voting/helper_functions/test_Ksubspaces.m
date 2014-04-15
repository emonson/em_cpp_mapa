% Complete rewrite of the test_gpca function to use the generalized data
% generation function instead.
clc;
clear;
close all;
addpath helper_functions


DEBUG = 1;

% Uncomment the line for the test(s) you wish to perform.
testMethod=1;
roundNumber = 100;

seedTimeStamp = sum(100*clock); %% Generate a new random seed.
rand('state',seedTimeStamp);
randn('state', seedTimeStamp);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The default case, which happens to be our favorite.
if testMethod==1
    % Generate the data samples
    ambientDimension = 3;
    groupCount = 3; 
    subspaceDimensions = [2 2 1];
    
    errorStat = 0;
    for roundIndex=1:roundNumber
    [X, aprioriSampleLabels, aprioriGroupBases, aprioriBasisDimensions] = generate_samples('ambientSpaceDimension', ambientDimension,...
        'basisDimensions', subspaceDimensions,...
        'noiseLevel', .02,...
        'minimumSubspaceAngle', pi/4);
    
    % Perform Ksubspaces on the data
    [sampleLabels, groupBases] = Ksubspaces(X, subspaceDimensions);
    
    
    % Reorder the bases and sample labels so that they are likely to agree with
    % the original ones.
    [bestMapping, errorProbability, groupError] = relabel_samples(aprioriSampleLabels, sampleLabels);
    errorProbability
    errorStat = errorStat + errorProbability;
    end
    errorStat = errorStat / roundNumber
    
    inverseMapping(bestMapping) = 1:length(bestMapping);
    fixedSampleLabels = bestMapping(sampleLabels);
    for subspaceIndex=1:groupCount
        fixedBases{subspaceIndex} = groupBases{inverseMapping(subspaceIndex)};
    end

    % Plot the gpca results
    if ambientDimension<=3
        figureNumber = figure();
        subplot(1,2,1);
        plot_data(X, aprioriSampleLabels);
        hold on;
        plot_subspaces(X, aprioriSampleLabels, aprioriGroupBases, aprioriBasisDimensions);
        hold off;
        title('Apriori Data');
        set(0,'CurrentFigure',figureNumber); % Change back to the correct figure without changing its visibility.
        subplot(1,2,2);
        plot_data(X, fixedSampleLabels);
        hold on;
        plot_subspaces(X, fixedSampleLabels, fixedBases, subspaceDimensions);
        hold off;
        title('Final Segmentation')
    end
end
