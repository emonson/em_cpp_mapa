function [X, sampleLabels, groupBases] = generate_samples_fast(ambientSpaceDimension, basisDimensions)
% [X, sampleLabels, groupBases, basisDimensions] = generate_samples_fast(ambientSpaceDimension, basisDimensions)
%
% Function to generate data to generate veronese map orders quickly.

BASE_SAMPLE_COUNT = 20;
groupCount = length(basisDimensions);

dimensionSampleCounts = zeros(1,ambientSpaceDimension-1);

veroneseSpaceDimension = nchoosek(groupCount+ambientSpaceDimension-1, groupCount);

% New Heuristic: The total number of points should be twice the dimension of the veronese
% map, and proportional to the dimension of the group.
samplesPerDimension = 2*veroneseSpaceDimension / sum(basisDimensions);
groupSizes = ceil(basisDimensions .* samplesPerDimension);

minimumSubspaceAngle = 20*pi/180;

% Initialization of the group Bases.
groupBases = repmat(eye(ambientSpaceDimension), [1 1 groupCount]);

minimumSubspaceAngleViolated = true;
maxIterations = 20;
iterationIndex = 1;
while(minimumSubspaceAngleViolated && maxIterations <= 20),

    % Generate the data for each group
    groupSamples = cell(1, groupCount);
    sampleLabels = zeros(1,sum(groupSizes));
    X = zeros(ambientSpaceDimension,sum(groupSizes));
    sampleLabelIndex = 1;
    for groupIndex = 1:groupCount,
        currentGroup = zeros(ambientSpaceDimension, groupSizes(groupIndex));
        currentGroup(1:basisDimensions(groupIndex),:) = rand_uniform_on_hypersphere(basisDimensions(groupIndex), groupSizes(groupIndex));

        % Rotate the group to an arbitrary orientation.
        newGroupOrientation = rand_special_orthogonal(ambientSpaceDimension);
        currentGroup =  newGroupOrientation * currentGroup;
        groupBases(:,:,groupIndex) = newGroupOrientation;

        % Combine the data and generate labels for the samples.
        X(:, sampleLabelIndex:sampleLabelIndex+groupSizes(groupIndex)-1) = currentGroup;
        sampleLabels(sampleLabelIndex:sampleLabelIndex+groupSizes(groupIndex)-1) = groupIndex;
        sampleLabelIndex = sampleLabelIndex + groupSizes(groupIndex);
    end %for

    % Use the group Bases to determine if the data that was generated
    % satisfies the minimum subspace angle criteria.
    smallestPairwiseAngle = pi/2;
    for firstGroupIndex = 1:groupCount,
        for secondGroupIndex = firstGroupIndex+1:groupCount,
            firstGroupBases = groupBases(:,1:basisDimensions(firstGroupIndex),firstGroupIndex);
            secondGroupBases = groupBases(:,1:basisDimensions(secondGroupIndex), secondGroupIndex);
            currentPairwiseAngle = subspace_gpca(firstGroupBases, secondGroupBases);
            if currentPairwiseAngle < smallestPairwiseAngle,
                smallestPairwiseAngle = currentPairwiseAngle;
            end % if
        end % for
    end % for

    if smallestPairwiseAngle >= minimumSubspaceAngle,
        finalX = X;
        finalGroupBases = groupBases;
        minimumSubspaceAngleViolated = false;
    else
        iterationIndex = iterationIndex + 1;
    end % if

end %while
if minimumSubspaceAngleViolated
    error(['Unable to find a data set that enforces the minimum angle after ' num2str(maxIterations) ' iterations. Try lowering ''minimumSubspaceAngle''']);
end

X = finalX;
groupBases = finalGroupBases;


