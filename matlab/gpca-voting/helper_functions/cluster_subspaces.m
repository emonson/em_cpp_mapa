function sampleLabels = cluster_subspaces(subspaces, subspaceDimensions, energyThreshold);
% cluster_subspaces is a function to group data vectors based on the
% subspace dimension assumption "subspaceDimensions".

% Step 1: recursively select a global energy threshold such that the
% cut-off of the singluar values match the subspaceDimensions assumption
% the best. 
[ambientDimension, charDimension, sampleNumber]= size(subspaces);
normalizedSubspaces=cell(1,sampleNumber);
sampleSitsInDimension=zeros(1,sampleNumber);

% Step 1.1: If the subspaceDimensions are all the same, then we don't need a energy
% cut.
mixedDimension = sum(subspaceDimensions~=subspaceDimensions(1));
if mixedDimension==0
    % Direct cut null spaces to their correct dimension
    dimensionClassNumber = 1;
    dimensionClasses = subspaceDimensions(1);
    for sampleIndex=1:sampleNumber
        sampleSitsInDimension(sampleIndex)=dimensionClasses;
        [U,S,V]=svds(subspaces(:,:,sampleIndex),dimensionClasses);
        normalizedSubspaces{sampleIndex}=U;
    end
else
    % Step 1.2: cut by an optimal global energy threshold.
    dimensionClassNumber = 0;
    for index=1:ambientDimension-1
        if sum(subspaceDimensions==index)>0
            dimensionClassNumber = dimensionClassNumber + 1;
            dimensionClasses(dimensionClassNumber) = index;
        end
    end
    
    for sampleIndex=1:sampleNumber
        [U,S,V]=svd(subspaces(:,:,sampleIndex));
        % Compute energy
        singularValues=diag(S);
        totalEnergy(sampleIndex) = sum(abs(singularValues).^2);
        dimensionAssigned = false;
        for dimensionClassIndex=1:dimensionClassNumber-1
            partialEnergy = sum(abs(singularValues(1:dimensionClasses(dimensionClassIndex))).^2);
            if partialEnergy/totalEnergy(sampleIndex) >= energyThreshold
                sampleSitsInDimension(sampleIndex) = dimensionClasses(dimensionClassIndex);
                normalizedSubspaces{sampleIndex}=U(:,1:sampleSitsInDimension(sampleIndex));
                dimensionAssigned = true;
                break;
            end
        end
        if dimensionAssigned == false
            % Just assign the maximal value
            sampleSitsInDimension(sampleIndex) = dimensionClasses(end);
            normalizedSubspaces{sampleIndex}=U(:,1:end-1);
        end
    end
end

% Step 2: cluster groups within each class with the same dimension.
labelBound = 0;
for dimensionClassIndex=1:dimensionClassNumber
    % The number of subspaces with the same dimension in the class
    clusterNumber = sum(subspaceDimensions==dimensionClasses(dimensionClassIndex));
    
    
    % Regroup samples in the same class
    subsampleIndices = (sampleSitsInDimension==dimensionClasses(dimensionClassIndex));
    if isempty(subsampleIndices)
        error('Encounter empty groups. Clustering step failed.');
    end
    subsampleMapping = find(subsampleIndices);
    if clusterNumber>1
        for index=1:length(subsampleMapping)
            subsampleSubspaces{index}=normalizedSubspaces{subsampleMapping(index)};
        end

        % invoke K_means_subspaces_fixed_dimension with number of clusters
        % given by clusterNumber.
        subsampleLabels = K_means_subspaces_fixed_dimension(subsampleSubspaces,clusterNumber);
    else
        subsampleLabels = ones(1,length(subsampleMapping));
    end
    subsampleLabels = subsampleLabels + labelBound;
    labelBound = labelBound + clusterNumber;
    
    % Map labels back to original samples
    sampleLabels(subsampleIndices) = subsampleLabels;
end
