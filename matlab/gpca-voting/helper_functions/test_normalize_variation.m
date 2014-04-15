groupCount = 3;
subspaceDimensions = 2*ones(1,groupCount);
[X, aprioriSampleLabels, aprioriGroupBases, aprioriBasisDimensions] = generate_samples('ambientSpaceDimension', 3,...
    'basisDimensions', subspaceDimensions,...
    'noiseLevel', .02,...
    'minimumSubspaceAngle', pi/4);

    figureNumber = figure();
    subplot(1,2,1);
    plot_data(X, aprioriSampleLabels);
    hold on;
    plot_subspaces(X, aprioriSampleLabels, aprioriGroupBases, aprioriBasisDimensions);
    hold off;

[X, H]=normalize_variation(X);

    subplot(1,2,2);
    plot_data(X, aprioriSampleLabels);
    hold on;
    plot_subspaces(X, aprioriSampleLabels, aprioriGroupBases, aprioriBasisDimensions);
    hold off;