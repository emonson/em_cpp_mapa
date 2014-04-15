function sampleLabels = intersection_reclassification(X, sampleLabels,subspaceNumber);

% Set constant values
K_NEAREST_NEIGHBOR = 1;
[ambientDimension, sampleNumber]=size(X);

method = K_NEAREST_NEIGHBOR;

if method == K_NEAREST_NEIGHBOR
    KConstant = 5;
    
    % Create distance matrix
    distanceMatrix = zeros(sampleNumber, sampleNumber);
    for sampleIndex=1:sampleNumber
        for index=sampleIndex+1:sampleNumber
            distanceMatrix(sampleIndex,index)=norm(X(:,sampleIndex)-X(:,index));
        end
        % Assign an artifial value for the sample itself so that it will
        % never be its own neighbor.
        distanceMatrix(sampleIndex, sampleIndex)=inf;
    end
    distanceMatrix = distanceMatrix + distanceMatrix.';
    
    % For each sample, find the nearest KConstant neighbors
    for sampleIndex=1:sampleNumber
        [ignored, index]= sort(distanceMatrix(:,sampleIndex));
        neighborLabels=sampleLabels(index(1:KConstant));
        for subspaceIndex=1:subspaceNumber
            neighborWeight = sum(neighborLabels==subspaceIndex);
            if neighborWeight>KConstant/2
                sampleLabels(sampleIndex)=subspaceIndex;
                break;
            end
        end        
    end
end

