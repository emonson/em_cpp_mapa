function [sampleLabels, groupBases, returnStatus]=Ksubspaces(X, subspaceDimensions, varargin)

% Assign constant values
maxLoopValue = 100;

[ambientDimension sampleNumber]= size(X);
subspaceNumber = length(subspaceDimensions);
returnStatus=0;

% Parse the optional inputs.
if (mod(length(varargin), 2) ~= 0 ),
    error(['Extra Parameters passed to the function ''' mfilename ''' must be passed in pairs.']);
end
parameterCount = length(varargin)/2;
stopThreshold = 0;
basesInitialized = false;
for parameterIndex = 1:parameterCount,
    parameterName = varargin{parameterIndex*2 - 1};
    parameterValue = varargin{parameterIndex*2};
    switch lower(parameterName)
        case 'initialbases'
            groupBases=parameterValue;
            basesInitialized = true;
        case 'stopthreshold'
            stopThreshold = parameterValue;
        otherwise
            error('Unknown parameter input.');
    end
end

% Initialize bases
if basesInitialized == false
    sampleLabels = ceil(rand(1,sampleNumber)*subspaceNumber);
    groupBases = subspace_bases(X, subspaceNumber, sampleLabels, subspaceDimensions);
end

if stopThreshold==0
    % Assign default stop threshold
    stopThreshold = 1e-6;
end

% Start iteration
loopCount = 0;
while 1
    loopCount = loopCount + 1;
    
    % Group samples based on distances
    for subspaceIndex=1:subspaceNumber
        distances(subspaceIndex,:)=point_to_space_distance(X,groupBases{subspaceIndex});
    end
    [ignored, sampleLabels]=min(distances);
    isClassEmpty = false;
    for subspaceIndex=1:subspaceNumber
        if sum(sampleLabels==subspaceIndex)==0
            isClassEmpty = true;
            break;
        end
    end
    
    if isClassEmpty==false
        % Estimate new bases
        newBases=subspace_bases(X, subspaceNumber, sampleLabels, subspaceDimensions);

        % Test stop threshold
        angleChange = 0;
        for subspaceIndex=1:subspaceNumber
            angleChange = angleChange + subspace_angle(groupBases{subspaceIndex}, newBases{subspaceIndex});
        end
        groupBases = newBases;
        if angleChange<stopThreshold
            break;
        end
    else
        % Restart the iteration but cut the loop limit by half
        loopCount = 0;
        maxLoopValue = floor(maxLoopValue/2);
        sampleLabels = ceil(rand(1,sampleNumber)*subspaceNumber);
        groupBases = subspace_bases(X, subspaceNumber, sampleLabels, subspaceDimensions);
    end
    
    % Iteration Upper Limit
    if loopCount>maxLoopValue
        warning('K-subspaces may not converge.');
        returnStatus=1;
        break;
    end
end

% Finally, have to test the empty class exception
if isClassEmpty==true
    warning('K-subspaces failed. The result contains empty classes.');
    sampleLabels=[];
    groupBases=[];
    returnStatus = -1;
end