function [sampleLabels, subspaceBases, returnStatus]=gpca_dimensions_specified(X, subspaceDimensions, varargin)

addpath helper_functions

% Set Constants
SVD_NULLSPACE=1;
FISHER_NULLSPACE=0;
RAYLEIGHQUOTIENT_EPSILON = 0.0001;

[ambientDimension sampleCount]= size(X);
subspaceCount = length(subspaceDimensions);

if (max(subspaceDimensions)>=ambientDimension) || (min(subspaceDimensions)<=0)
    error('Illegal input for subspace dimension(s).');
end

if subspaceCount==1
    % Just one subspace, invoke PCA and quit
    sampleLabels=ones(1,sampleCount);
    returnStatus = 0;
    [subspaceBases{1},SS,VV] = svds(X,subspaceDimensions(1));
    return;
end

% Set default parameter values
angleTolerance = 0.1;
dataNormalization = true;
postOptimization = true;
nullSpaceMethod = FISHER_NULLSPACE;

% Parse the optional inputs.
if (mod(length(varargin), 2) ~= 0 ),
    error(['Extra Parameters passed to the function ''' mfilename ''' must be passed in pairs.']);
end
parameterCount = length(varargin)/2;

parameterInitialized = false;
for parameterIndex = 1:parameterCount,
    parameterName = varargin{parameterIndex*2 - 1};
    parameterValue = varargin{parameterIndex*2};
    switch lower(parameterName)
        case 'angletolerance'
            angleTolerance = parameterValue;
        case 'datanormalization'
            dataNormalization = parameterValue;
        case 'postoptimization'
            postOptimization = parameterValue;
        case 'nullspacemethod'
            nullSpaceMethod = parameterValue;
    end
end

% Normalize data vector variations
if dataNormalization
    [X, subspaceVariances]=normalize_variance(X);
end

% Re-order the samples based on the Mahalanobis distance.
sampleDistances=mahalanobis_distance(X');
[ignored, sampleDistanceIndex] = sort(sampleDistances,'descend');
X=X(:,sampleDistanceIndex);
inverseDistanceIndex(sampleDistanceIndex)=1:sampleCount;

% Start GPCA main function
[veroneseMap, veroneseMapDerivative] = generate_veronese_maps(X, subspaceCount, 'single');
veroneseMapDimension = size(veroneseMap,1);

% Find corresponding null space dimension from the Hilbert function
% constraint
if ambientDimension==2
    charDimension = 1;
else
    charDimension = Hilbert_function(ambientDimension,subspaceDimensions);
end

% Compute null space coefficients
if nullSpaceMethod==FISHER_NULLSPACE
    % Compute null space by Rayleigh quotient
    A = zeros(veroneseMapDimension, veroneseMapDimension);
    B = A;
    for sampleIndex = 1: sampleCount
        A = A + veroneseMap(:,sampleIndex) * veroneseMap(:,sampleIndex).';
        B = B + veroneseMapDerivative(:,:,sampleIndex) * veroneseMapDerivative(:,:,sampleIndex).';
    end
    
    A = A+RAYLEIGHQUOTIENT_EPSILON*eye(veroneseMapDimension);
    [eigenvectors, D] = eig(A, B, 'qz');
    [ignore eigenvalueIndex] = sort(diag(D),1,'descend');
    
    nullSpaceCoefficients = eigenvectors(:,eigenvalueIndex(end+1-charDimension:end));
    % normalize coefficients
    for index=1:charDimension
        nullSpaceCoefficients(:,index)= nullSpaceCoefficients(:,index)/norm(nullSpaceCoefficients(:,index));
    end
else
    % Compute null space by SVD
    [U,S,V] = svd(veroneseMap');
    nullSpaceCoefficients = V(:,end+1-charDimension:end);
end

% Evaluate null spaces
nullSpaces=zeros(ambientDimension,charDimension,sampleCount);
for dimensionIndex = 1:ambientDimension,
    nullSpaces(dimensionIndex, :,:) = nullSpaceCoefficients' * squeeze(veroneseMapDerivative(:,dimensionIndex,:));
end

% Vote subspaces based on the dimension assumption
[subspaceBases, sampleLabels] = subspace_voting(nullSpaces, ambientDimension*ones(1,subspaceCount)-subspaceDimensions, angleTolerance);

if isempty(sampleLabels)
    returnStatus = -1;
    return;
end

% Post-optimization
if postOptimization
    [sampleLabels, subspaceBases, returnStatus]=Ksubspaces(X, subspaceDimensions, 'initialBases',subspaceBases);
    
        % Finally, pose-processing the subspaceBases
    if dataNormalization
        X = normalize_variance(X,subspaceVariances);
        for subspaceIndex=1:subspaceCount
            subspaceX=X(:,find(sampleLabels==subspaceIndex));
            [U,S,V]=svds(subspaceX, subspaceDimensions(subspaceIndex));
            subspaceBases{subspaceIndex} = U;
        end
    end
else
    for sampleIndex=1:sampleCount
        if sampleLabels(sampleIndex)==0
            % Classify missing samples in subspace_voting process
            for subspaceIndex=1:subspaceCount
                distances(subspaceIndex)=point_to_space_distance(X(:,sampleIndex),subspaceBases{subspaceIndex});
            end
            [ignored,sampleLabels(sampleIndex)]=min(distances);
        end
    end

    % Finally, pose-processing the subspaceBases
    if dataNormalization
        X = normalize_variance(X,subspaceVariances);
    end
    for subspaceIndex=1:subspaceCount
        subspaceX=X(:,find(sampleLabels==subspaceIndex));
        [U,S,V]=svds(subspaceX, subspaceDimensions(subspaceIndex));
        subspaceBases{subspaceIndex} = U;
    end

    returnStatus = 0;
end

% Change the sample order from Mahalanobis order to the original
sampleLabels = sampleLabels(inverseDistanceIndex);
