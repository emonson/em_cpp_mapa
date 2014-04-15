function [sampleLabels, groupBases, returnStatus]=EM_subspace(X, subspaceDimensions, varargin)

% Assign constant values
maxLoopValue = 100;

[ambientDimension sampleNumber]= size(X);
subspaceNumber = length(subspaceDimensions);
returnStatus = 0;

% Parse the optional inputs.
if (mod(length(varargin), 2) ~= 0 ),
    error(['Extra Parameters passed to the function ''' mfilename ''' must be passed in pairs.']);
end
parameterCount = length(varargin)/2;
stopThreshold = 0;
parameterInitialized = false;
for parameterIndex = 1:parameterCount,
    parameterName = varargin{parameterIndex*2 - 1};
    parameterValue = varargin{parameterIndex*2};
    switch lower(parameterName)
        case 'initialdualbases'
            dualBases=parameterValue;
            parameterInitialized = true;
        case 'initialbases'
            groupBases=parameterValue;
            dualBases=cell(1,subspaceNumber);
            for subspaceIndex=1:subspaceNumber
                dualBases{subspaceIndex}=null(groupBases{subspaceIndex}');
            end
            parameterInitialized = true;
        case 'stopthreshold'
            stopThreshold = parameterValue;
        otherwise
            error('Unknown parameter input.');
    end
end

% Initialize probability parameters
Pi=1/subspaceNumber*ones(1,subspaceNumber);
Sigma2=ones(1,subspaceNumber);

% Initialize bases
if parameterInitialized == false
    % Randomly generate a set of subspace bases
    for subspaceIndex=1:subspaceNumber
        dualBases{subspaceIndex}=zeros(ambientDimension,subspaceDimensions(subspaceIndex));
        for dimensionIndex=1:ambientDimension-subspaceDimensions(subspaceIndex)
            vector = randn(ambientDimension,1);
            dualBases{subspaceIndex}(:,dimensionIndex) = vector/norm(vector);
        end
    end
end

if stopThreshold==0
    % Assign default stop threshold
    stopThreshold = 1e-6;
end

% Start iteration
loopCount = 0;
for sampleIndex=1:sampleNumber
    xxMatrix{sampleIndex}=X(:,sampleIndex)*X(:,sampleIndex).';
end
while 1
    loopCount = loopCount + 1;
    
    % Group samples based on distances
    
    for sampleIndex=1:sampleNumber
        xVector=X(:,sampleIndex);
        
        denominator = 0;
        for subspaceIndex=1:subspaceNumber
            p(subspaceIndex)=1/(2*pi*Sigma2(subspaceIndex))*exp(-(xVector.'*dualBases{subspaceIndex}*dualBases{subspaceIndex}.'*xVector)/(2*Sigma2(subspaceIndex)));
            denominator = denominator + p(subspaceIndex)*Pi(subspaceIndex);
        end
        
        for subspaceIndex=1:subspaceNumber
            W(subspaceIndex,sampleIndex)=Pi(subspaceIndex)*p(subspaceIndex)/denominator;
        end
    end
    sumW = sum(W,2);
    
    % Assign new membership
    [ignored, sampleLabels]=max(W);
    isClassEmpty = false;
    for subspaceIndex=1:subspaceNumber
        if sum(sampleLabels==subspaceIndex)==0
            isClassEmpty = true;
            break;
        end
    end
    
    % Update parameters
    if isClassEmpty==false
        angleChange = 0;
        for subspaceIndex=1:subspaceNumber
            weightedMatrix = 0;
            for sampleIndex=1:sampleNumber
                weightedMatrix = weightedMatrix + W(subspaceIndex,sampleIndex)*xxMatrix{sampleIndex};
            end
            [U,S,V]=svd(weightedMatrix);
            newBases{subspaceIndex} = V(:,end+1-(ambientDimension-subspaceDimensions(subspaceIndex)):end);
            angleChange = angleChange + subspace_angle(dualBases{subspaceIndex}, newBases{subspaceIndex});

            Pi(subspaceIndex) = sumW(subspaceIndex)/sampleNumber;

            weightedMatrix = 0;
            for sampleIndex=1:sampleNumber
                weightedMatrix = weightedMatrix + W(subspaceIndex,sampleIndex)*norm(newBases{subspaceIndex}.'*X(:,sampleIndex))^2;
            end
            Sigma2(subspaceIndex) = weightedMatrix/(ambientDimension-subspaceDimensions(subspaceIndex))/sumW(subspaceIndex);
        end

        dualBases = newBases;

        % Test stop threshold
        if angleChange<stopThreshold
            break;
        end
    else
        % Randomly generate another set of bases
        for subspaceIndex=1:subspaceNumber
            dualBases{subspaceIndex}=zeros(ambientDimension,subspaceDimensions(subspaceIndex));
            for dimensionIndex=1:ambientDimension-subspaceDimensions(subspaceIndex)
                vector = randn(ambientDimension,1);
                dualBases{subspaceIndex}(:,dimensionIndex) = vector/norm(vector);
            end
        end
        
        loopCount = 0;
        maxLoopValue = floor(maxLoopValue/2);
    end
    
    if loopCount>maxLoopValue
        warning('EM iteration may not converge.');
        returnStatus = 1;
        break;
    end
end

% Have to test the empty class exception
if isClassEmpty
    warning('EM failed. The result contains empty classes.');
    sampleLabels=[];
    groupBases=[];
    returnStatus = -1;
else
    % Compute groupBases from dualBases
    groupBases=cell(1,subspaceNumber);
    for subspaceIndex=1:subspaceNumber
        groupBases{subspaceIndex}=null(dualBases{subspaceIndex}');
    end
end