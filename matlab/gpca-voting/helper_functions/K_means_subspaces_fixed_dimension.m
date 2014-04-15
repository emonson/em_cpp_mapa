function sampleLabels = K_means_subspaces_fixed_dimension(subspaces, clusterNumber);
% Use K means technique to cluster subspaces into groups. The distance
% metric is the space angle between two subspaces.

% Set constant values
sampleNumber = size(subspaces,2);
subspaceDimension = size(subspaces{1},2);
sampleLabels = ones(1,sampleNumber);


% To initialize the algorithm, we need to randomly assign several cluster
% centers.
for clusterIndex=1:clusterNumber
   clusterCenterIndex(clusterIndex) = ceil(sampleNumber*rand());
   while (clusterIndex>1) && (sum(clusterCenterIndex(1:clusterIndex-1)==clusterCenterIndex(clusterIndex))>0)
       clusterCenterIndex(clusterIndex) = ceil(sampleNumber*rand());
   end
   clusterCenter{clusterIndex}=subspaces{clusterCenterIndex(clusterIndex)};
end

% Recursively update the cluster center
stopFlag=false;
loopCount = 0;
while stopFlag==false
    loopCount = loopCount + 1;
    
    % Step 1. Cluster subspaces based on the cluster centers
    dimensionCount=zeros(1,clusterNumber);
    matrix=cell(1,clusterNumber);
    for sampleIndex=1:sampleNumber
        for clusterIndex=1:clusterNumber
            distance(clusterIndex) = subspace_angle(subspaces{sampleIndex},clusterCenter{clusterIndex});
        end
        [ignored,nearestCenter] = min(distance);
        sampleLabels(sampleIndex)=nearestCenter;
        matrix{sampleLabels(sampleIndex)}(:,dimensionCount(nearestCenter)+1:...
        dimensionCount(nearestCenter)+subspaceDimension)=subspaces{sampleIndex};
        dimensionCount(nearestCenter) = dimensionCount(nearestCenter) + subspaceDimension;
    end
    
    % Step 2. Compute new cluster centers in clusters
    angleChanges = 0;
    for clusterIndex=1:clusterNumber
        if dimensionCount(clusterIndex)==0
            error('Encounter empty groups. Kmeans algorithm failed.');
        end
        [U,S,V]=svds(matrix{clusterIndex}, subspaceDimension);
        angleChanges = angleChanges + subspace_angle(U,clusterCenter{clusterIndex});
        clusterCenter{clusterIndex}=U;
    end
    
    % Step 3. Test stop conditions
    if angleChanges<1e-6
        stopFlag = true;
    end
    if loopCount>100
        stopFlag = true;
    end
end