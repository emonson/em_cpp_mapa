function newBases=subspace_bases(X, subspaceNumber, sampleLabels, subspaceDimensions)

[ambientDimension, sampleNumber]=size(X);

for subspaceIndex=1:subspaceNumber
    groupX=X(:,sampleLabels==subspaceIndex);
    groupSize=size(groupX,2);
    if groupSize==0
        % It is an empty group. 
        error('There are empty classes in sampleLables. The function subspace_bases cannot proceed.');
    else
        [U,S,V]=svds(groupX,min(subspaceDimensions(subspaceIndex),groupSize));
        sDiag=diag(S);
        uSize=sum(sDiag>0);
        newBases{subspaceIndex}(:,1:uSize)=U(:,1:uSize);
        for dimensionIndex=uSize+1:subspaceDimensions(subspaceIndex)
            vector = randn(ambientDimension,1);
            newBases{subspaceIndex}(:,dimensionIndex) = vector/norm(vector);
        end
    end
end