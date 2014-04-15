function [X, sampleLabels, groupBases, basisDimensions] = generate_samples(varargin)
% [X, sampleLabels, groupBases, basisDimensions] = generate_samples(varargin)
%
% The gpca data generation function to end all test data generation
% functions.
%
% Inputs: (Everything is Optional)
%
% ambientSpaceDimension:   (Default is 3) 
%
% basisDimensions:   A vector the length of the number of groups,
% containing
%                    the dimensions of the groups
%
% basisDimensionType:   If basisDimensions is not specified, this is one of
%                       'hyperplanes' 'lines' 'oneOfEachDimension'
%
% groupDistributionType:   One of 'uniformCube', 'uniformSphere', 'gaussian',
%                           'uniformSphereSurface'
%
% groupDistributionStandardDeviations:  The standard
%                                       deviation of the distance of the
%                                       points to the origin.  Defaults
%                                       to .5 for every group so
%                                       uniformly distributed data
%                                       "fills" the unit sphere.
%                                       Allows you to have some
%                                       groups extend farther in
%                                       the ambient space than others.
%
%       WARNING: STD NOT BING COMPUTED PROPERLY FOR ALL DISTRIBUTIONS AT
%       MOMENT.
%
%
% groupSizes:   A vector
%               specifying the number of
%               points in each
%               group.
%
% noiseType: One of 'multiplicative' or 'additive'
%            
% noiseStatistic: One of 'uniform' or 'gaussian'
%
% noiseLevel: The standard deviation of the noise.
%
% scrambleOrder: One of true or false (i.e. 1 or 0, not the string)
%
% minimumSubspaceAngle: If specified, will try to 
%                       enforces a worst case angle between
%                       any two subspaces.
%
% Note: The first group is left aligned with the low dimension axes. This
%       i.e. if you are displaying a plane and two lines, putting the plane
%       first should plot nicely.

ALIGN_FIRST_GROUP = true; % Decide whether or not to align the first group with the axes.  
%This can be easier to display, but the presence of zeros in the array can
%make some of the computations later on more difficult (i.e. can't take log)

BASE_SAMPLE_COUNT = 40; % The base number of points to have for a one dimensional group.

% Set up default values below:
outlierPercentage = 0;
outlierNumber = 0;
ambientSpaceDimension = 3;
basisDimensions = [2 1 1];                  % One Plane and two lines, our favorite three dimensional test case.
groupDistributionType = 'uniformSphere'; 
noiseType = 'additive';
noiseStatistic = 'gaussian';
noiseLevel = 0;                             % By default, the sample is noise-free
scrambleOrder = false;                      % By default, do not scramble samples.
minimumSubspaceAngle = 0;                   % By default, allow subspaces that are subsets of each other.
avoidIntersection = false;                  % By default, generate samples on subspace intersections
maxIterations = 5000;
offsetMagnitude = 2;
gaussianDeviation = [];
isAffine=false;

% Parse the optional inputs. 
if mod(length(varargin), 2) ~= 0,
    error(['Extra Parameters passed to the function ''' mfilename ''' must be passed in pairs.']);
end
parameterCount = length(varargin)/2;
for parameterIndex = 1:parameterCount,
    parameterName = varargin{parameterIndex*2 - 1};
    parameterValue = varargin{parameterIndex*2};
    switch lower(parameterName)
        case 'isaffine'
            isAffine = parameterValue;
            if parameterValue~=false & parameterValue~=true
                error('The parameter isAffine must be a boolean variable.');
            end
        case 'ambientspacedimension'
            ambientSpaceDimension = parameterValue;
            if ambientSpaceDimension < 2 || ~isnumeric(ambientSpaceDimension),
                error('The dimension of the ambient space should be an integer larger than one.');
            end
        case 'basisdimensions'
            basisDimensions = parameterValue;
        case 'gaussiandeviation'
            gaussianDeviation = parameterValue;
        case 'offsetmagnitude'
            offsetMagnitude = parameterValue;
        case 'basisDimensionType'
            basisDimensionType = lower(parameterValue);
            if strcmpi(basisDimensionType, 'hyperplanes')
                basisDimensionType = 'hyperplanes';
            elseif strcmpi(basisDimensionType, 'lines')
                basisDimensionType = 'lines';
            elseif strcmpi(basisDimensionType, 'oneOfEachDimension')
                basisDimensionType = 'oneOfEachDimension';
            else
                error('basisDimensionType must be one of ''hyperplanes'' ''lines'' ''oneOfEachDimension''.')
            end
        case 'groupdistributiontype'
            groupDistributionType = lower(parameterValue);
            if strcmpi(groupDistributionType, 'uniformcube')
                groupDistributionType = 'uniformCube';
            elseif strcmpi(groupDistributionType, 'uniformsphere'),
                groupDistributionType = 'uniformSphere';
            elseif strcmpi(groupDistributionType, 'gaussian')
                groupDistributionType = 'gaussian';
            else
                error('groupDistributionType must be one of ''uniformcube'', ''uniformSphere'', ''gaussian''.')
            end
        case 'groupsizes'    
            groupSizes = parameterValue;          
        case 'noisetype'
            noiseType = parameterValue;
            if strcmpi(noiseType, 'multiplicative')
                noiseType = 'multiplicative';
            elseif strcmpi(noiseType, 'additive')
                noiseType = 'additive';
            else
                error('noiseType must be one of ''multiplicative'' or ''additive''.')
            end
        case 'noisestatistic'
            noiseStatistic = parameterValue;
            if strcmpi(noiseStatistic, 'uniform'),
                noiseStatistic = 'uniform';
            elseif strcmpi(noiseStatistic, 'gaussian')
                noiseStatistic = 'gaussian';
            else
                error('noiseStatistic must be one of ''uniform'' or ''gaussian''.')
            end
        case 'noiselevel'
            noiseLevel = parameterValue;
            if noiseLevel < 0 || ~isnumeric(noiseLevel),
                error('noiseLevel should be a positive or zero numeric value.')
            end
        case 'scrambleorder'
            scrambleOrder = parameterValue;
            if ischar(scrambleOrder),
                switch lower(scrambleOrder)
                    case 'true'
                        scrambleOrder = true;
                    case 'false'
                        scrambleOrder = false;
                    otherwise
                        error('Value for scrambleOrder should be a logical true or false')
                end
            end
        case 'minimumsubspaceangle'
            minimumSubspaceAngle = parameterValue;
            if minimumSubspaceAngle < 0 || minimumSubspaceAngle > pi/2,
                error('Value for minimumSubspaceAngle must be between 0 and pi/2 radians.')
            end
        case 'outlierpercentage'
            outlierPercentage = parameterValue;
            if outlierPercentage<0 || outlierPercentage>1
                error('Outlier Percentage parameter is out of bound (between 0 and 1).');
            end
        case 'outliernumber'
            outlierNumber = parameterValue;
            if outlierNumber<0
                error('Outlier Number parameter is out of bound (>0).');
            end
        case 'avoidintersection'
            avoidIntersection = parameterValue;
            if avoidIntersection~=true && avoidIntersection~=false
                error('Avoid Intersection parameter is out of bound (True or False).');
            end
        otherwise 
            error(['Sorry, the parameter ''' parameterName ''' is not recognized by the function ''' mfilename '''.']);
    end
end

if exist('basisDimensionType','var') && exist('basisDimensions','var'),
    warning('It is not necessary to specify both basisDimensionType and basisDimensions');
elseif exist('basisDimensionType','var') && ~exist('basisDimensions','var'),    
    switch basisDimensionType
        case 'hyperplanes'
            basisDimensions = (ambientSpaceDimension - 1)*ones(1,ambientSpaceDimension); % One hyperplane for each dimension
        case 'lines'
            basisDimensions = ones(1,ambientSpaceDimension); % One line for each dimension
        case 'oneOfEachDimension'
            basisDimensions = 1:(ambientSpaceDimension - 1); % One of each dimension up to the hyperplane case
    end
end
groupCount = length(basisDimensions);
if isempty(gaussianDeviation)
    for index=1:groupCount
        gaussianDeviation{index} = eye(ambientSpaceDimension, ambientSpaceDimension);
    end
else
    if length(gaussianDeviation)~=groupCount
        error('The Gaussian deviation matrices do not match the groupCount number.');
    end
end

if strcmp(groupDistributionType, 'gaussian') 
    if max(basisDimensions) > ambientSpaceDimension
        error('for Gaussian models, group dimensions must be greater than or equal to the ambient dimension.')
    end
elseif max(basisDimensions) >= ambientSpaceDimension
    error('for subspace models, group dimensions must correspond to proper subspaces of the ambient space.')
end
if min(basisDimensions) <= 0,
    error('for generate_samples.m, the group dimensions must be greater than zero.')
end
if isAffine & ~avoidIntersection
    avoidIntersection = true;
    warning('The parameter isAffine overwrites the parameter avoidIntersection to be TRUE.');
end

if ~exist('groupSizes','var'),
    dimensionSampleCounts = zeros(1,ambientSpaceDimension-1);
    % Decide how many points groups of various dimensions should have.
    for dimensionIndex = 1:ambientSpaceDimension-1;
        % Try to preserve the arial density over dimensions.
        % This may be tragically incorrect, especialy since the area of the
        % distribution will depend on whether we are 'uniformSphere' or
        % 'uniformCube'
        switch groupDistributionType
            case {'gaussian' 'uniformCube'}
                dimensionSampleCounts(dimensionIndex) = ceil(BASE_SAMPLE_COUNT * 1);  
            case 'uniformSphere'
                % The number of points placed in the uniform
                % disc(generally, sphere) should be proportional to its
                % volume?
                dimensionSampleCounts(dimensionIndex) = ceil(BASE_SAMPLE_COUNT * hypersphere_volume(dimensionIndex));
            case 'uniformSphereSurface'
                % The number of points points placed one the surface of the
                % sphere should be proportional to its area (i.e. volume of
                dimensionSampleCounts(dimensionIndex) = max(ceil(BASE_SAMPLE_COUNT * hypersphere_area(dimensionIndex)),BASE_SAMPLE_COUNT);
        end
    end
    groupSizes = dimensionSampleCounts(basisDimensions);
end

% Initialization of the group Bases.
groupBases = cell(1,groupCount);

minimumSubspaceAngleViolated = true;
iterationIndex = 0;
newGroupOrientation = cell(1,groupCount);
sampleNumber = 0;
% Generate the data for each group
sampleLabels = zeros(1,sum(groupSizes));
X = zeros(ambientSpaceDimension,sum(groupSizes));
while (minimumSubspaceAngleViolated && iterationIndex <= maxIterations)
    iterationIndex = iterationIndex + 1;

    if ~strcmp(groupDistributionType, 'gaussian')
        for groupIndex = 1:groupCount

            % Rotate the group to an arbitrary orientation.
            newGroupOrientation{groupIndex} = rand_special_orthogonal(ambientSpaceDimension);
            if ALIGN_FIRST_GROUP && groupIndex==1
                % Without loss of generality, leave the first group aligned with
                % the original axes.
                groupBases{groupIndex} = eye(ambientSpaceDimension, basisDimensions(groupIndex));
            else
                groupBases{groupIndex} = newGroupOrientation{groupIndex}(:,1:basisDimensions(groupIndex));
            end

        end %for

        % Use the group Bases to determine if the data that was generated
        % satisfies the minimum subspace angle criteria.
        smallestPairwiseAngle = pi/2;
        for firstGroupIndex = 1:groupCount,
            for secondGroupIndex = firstGroupIndex+1:groupCount,
                firstGroupBases = groupBases{firstGroupIndex};
                secondGroupBases = groupBases{secondGroupIndex};
                currentPairwiseAngle = subspace_angle(firstGroupBases, secondGroupBases);
                if currentPairwiseAngle < smallestPairwiseAngle,
                    smallestPairwiseAngle = currentPairwiseAngle;
                end % if
            end % for
        end % for
    else
        smallestPairwiseAngle = minimumSubspaceAngle;
    end
    
    if smallestPairwiseAngle >= minimumSubspaceAngle
        % Generate samples according to the bases
        for groupIndex = 1:groupCount
            % Generate the data for the current Group, aligned with the first
            % coordinate axes for now.
            currentGroup = zeros(ambientSpaceDimension, groupSizes(groupIndex));
            switch groupDistributionType
                case 'uniformCube'
                    currentGroup(1:basisDimensions(groupIndex),:) = (rand(basisDimensions(groupIndex), groupSizes(groupIndex)) - .5);
                case 'uniformSphere'
                    currentGroup(1:basisDimensions(groupIndex),:) = rand_uniform_inside_hypersphere(basisDimensions(groupIndex), groupSizes(groupIndex));
                case 'uniformSphereSurface'
                    currentGroup(1:basisDimensions(groupIndex),:) = rand_uniform_on_hypersphere(basisDimensions(groupIndex), groupSizes(groupIndex));
                case 'gaussian'
                    currentGroup(1:basisDimensions(groupIndex),:) = gaussianDeviation{groupIndex}*randn(basisDimensions(groupIndex), groupSizes(groupIndex));
            end

            if ~strcmp(groupDistributionType, 'gaussian')
                if ALIGN_FIRST_GROUP
                    % Without loss of generality, leave the first group aligned with
                    % the original axes.
                    if groupIndex ~= 1
                        currentGroup =  newGroupOrientation{groupIndex} * currentGroup;
                    end
                else
                    currentGroup =  newGroupOrientation{groupIndex} * currentGroup;
                end
            end
            
            % Combine the data and generate labels for the samples.
            X(:, sampleNumber+1:sampleNumber+groupSizes(groupIndex)) = currentGroup;
            sampleLabels(sampleNumber+1:sampleNumber+groupSizes(groupIndex)) = groupIndex;
            sampleNumber = sampleNumber + groupSizes(groupIndex);
        end
        
        minimumSubspaceAngleViolated = false;
    end % if
    
end %while

if minimumSubspaceAngleViolated
    error(['Unable to find a data set that enforces the minimum angle after ' num2str(maxIterations) ' iterations. Try lowering ''minimumSubspaceAngle''']);
end

% Normalize the sample magnitude
if ~strcmp(groupDistributionType,'gaussian')
    for sampleIndex=1:sampleNumber
        Xnorm(sampleIndex) = norm(X(:,sampleIndex));
    end
    X=X/max(Xnorm);
end

% Apply the noise function globally to the data.
switch noiseStatistic
    case 'uniform'
        noise = noiseLevel * 2*(rand(size(X)) - .5);
    case 'gaussian'
        noise = noiseLevel * randn(size(X));
end
switch noiseType
    case 'multiplicative'
        X = X .* noise;
    case 'additive'
        X = X + noise;
end

% Avoid Intersections
if avoidIntersection | isAffine
    
    % Randomly change the center of the sample cluster on each subspace
    % away from the origin. 
    % Notice: by default the magnitudes range in [-1, 1].
    for groupIndex=1:groupCount
        
        if isAffine==false
            % Constrain the offset center to be within the original subspace
            offset = 2*offsetMagnitude*(rand(basisDimensions(groupIndex),1)-0.5*ones(basisDimensions(groupIndex),1));
            offset = groupBases{groupIndex}*offset;
        else
            % Affine case, the center can be moved arbitrarily in space.
            offset = 2*offsetMagnitude*(rand(ambientSpaceDimension,1)-0.5*ones(ambientSpaceDimension,1));
        end
        subspaceIndices = find(sampleLabels==groupIndex);
        offsetCenter = repmat(offset,1, length(subspaceIndices));
        X(:,subspaceIndices) = X(:,subspaceIndices) + offsetCenter;
    end
else
    offsetMagnitude = 1;
end

% Add outliers
if outlierPercentage>0 || outlierNumber>0
    if outlierNumber==0
        outlierNumber = ceil(sampleNumber * outlierPercentage / (1-outlierPercentage));
    end
    
    if strcmp(groupDistributionType,'gaussian')
        % the samples have a gaussian distribution
        for sampleIndex=1:sampleNumber
            Xnorm(sampleIndex) = norm(X(:,sampleIndex));
        end
        outlierMagnitude=max(Xnorm);
    else
        % otherwise, they have been normalized with magnitude max == 1;
        outlierMagnitude = offsetMagnitude;
    end
    outliers = outlierMagnitude*2*(rand(ambientSpaceDimension, outlierNumber)-0.5*ones(ambientSpaceDimension, outlierNumber));
    X = [X outliers];
    sampleLabels = [sampleLabels -ones(1,outlierNumber)];
end

% Scramble the order of the data in the array, scrambling the labels
% correspondingly.
if scrambleOrder,
    dataPermutationVector = randperm(size(X, 2));
    X = X(:,dataPermutationVector);
    sampleLabels = sampleLabels(dataPermutationVector);
end
