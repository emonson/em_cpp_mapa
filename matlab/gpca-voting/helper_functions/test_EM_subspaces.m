% Complete rewrite of the test_gpca function to use the generalized data
% generation function instead.
clear;
close all;
addpath helper_functions
clc;

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

        % Plot the Apriori Segmentation
        if ambientDimension<=3

        end

        % Perform Ksubspaces on the data
        [sampleLabels, groupBases] = EM_subspace(X, subspaceDimensions);


        % Reorder the bases and sample labels so that they are likely to agree with
        % the original ones.
        [bestMapping, errorProbability, groupError] = relabel_samples(aprioriSampleLabels, sampleLabels);
        errorProbability
        errorStat = errorStat + errorProbability;
    end
    errorStat = errorStat/roundNumber
    
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if testMethod==2
    % Lines in R2
    groupCount = 3;

    % Generate the data set.
    [X, aprioriSampleLabels, aprioriGroupBases, aprioriBasisDimensions] = generate_samples('ambientSpaceDimension', 2,...
        'basisDimensions', ones(1,groupCount),...
        'noiseLevel', .008,...
        'minimumSubspaceAngle', pi/8);
    
    % Plot the Apriori Segmentation
    figureNumber = figure('Visible','off');
    subplot(1,2,1);
    plot_data(X, aprioriSampleLabels);
    hold on;
    plot_subspaces(X, aprioriSampleLabels, aprioriGroupBases, aprioriBasisDimensions);
    hold off;
    title('Apriori Data');

    % Perform GPCA on the data
    [sampleLabels, groupBases, basisDimensions] = gpca(X, 'groupCount', groupCount);

    % Reorder the bases and sample labels so that they are likely to agree with
    % the original ones.
    [bestMapping, errorProbability, groupError] = relabel_samples(aprioriSampleLabels, sampleLabels, groupCount);
    inverseMapping(bestMapping) = 1:length(bestMapping);
    fixedSampleLabels = bestMapping(sampleLabels);
    fixedBases = groupBases(:,:,inverseMapping);
    fixedBasisDimensions = basisDimensions(inverseMapping);

    % Plot the GPCA results
    set(0,'CurrentFigure',figureNumber); % Change back to the correct figure without changing its visibility.
    subplot(1,2,2);
    plot_data(X, fixedSampleLabels);
    hold on;
    plot_subspaces(X, fixedSampleLabels, fixedBases, fixedBasisDimensions);
    hold off;
    title('Final Segmentation')
    set(gcf, 'Visible', 'on');

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if testMethod==3
    % Lines or planes in R3.  Meant to excercise specified group, specified
    % dimension.
    groupCount = 4;
    groupDimension = 1;

    % Generate the data set.
    [X, aprioriSampleLabels, aprioriGroupBases, aprioriBasisDimensions] = generate_samples('ambientSpaceDimension', 3,...
        'basisDimensions', groupDimension*ones(1,groupCount),...
        'noiseLevel', .0,...
        'minimumSubspaceAngle', pi/180*20);
    % Plot the Apriori Segmentation
    figureNumber = figure;
    subplot(1,2,1);
    plot_data(X, aprioriSampleLabels);
    hold on;
    plot_subspaces(X, aprioriSampleLabels, aprioriGroupBases, aprioriBasisDimensions);
    hold off;
    title('Apriori Data');

    % Perform GPCA on the data
    [sampleLabels, groupBases, basisDimensions] = gpca(X, 'groupCount', 4);

    % Reorder the bases and sample labels so that they are likely to agree with
    % the original ones.
    [bestMapping, errorProbability, groupError] = relabel_samples(aprioriSampleLabels, sampleLabels, groupCount);
    inverseMapping(bestMapping) = 1:length(bestMapping);
    fixedSampleLabels = bestMapping(sampleLabels);
    fixedBases = groupBases(:,:,inverseMapping);
    fixedBasisDimensions = basisDimensions(inverseMapping);

    % Plot the gpca results
    figure(figureNumber)
    subplot(1,2,2);
    plot_data(X, fixedSampleLabels);
    hold on;
    plot_subspaces(X, fixedSampleLabels, fixedBases, fixedBasisDimensions);
    hold off;
    title('Final Segmentation')
end
