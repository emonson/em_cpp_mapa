function [subspaceBases, sampleLabels]= subspace_voting(subspaces, subspaceDimensions, angleTolerance)

MERGE_VOTES=true;

[ambientDimension, charDimension, sampleCount]= size(subspaces);

% Get voting classes
dimensionClassCount = 0;
for index=1:ambientDimension-1
    if sum(subspaceDimensions==index)>0
        dimensionClassCount = dimensionClassCount + 1;
        dimensionClasses(dimensionClassCount) = index;
    end
end

% Start voting
bases=cell(1,dimensionClassCount);
vote=cell(1,dimensionClassCount);
basisCount=zeros(1,dimensionClassCount);
sampleVotes=zeros(dimensionClassCount,sampleCount);
for sampleIndex=1:sampleCount
    [U,S,V]=svd(subspaces(:,:,sampleIndex));
    
    for classIndex=1:dimensionClassCount
        normalizedSubspaces=U(:,1:dimensionClasses(classIndex));
        % Register this basis
        if sampleIndex==1
            % the first vote in this class
            basisCount(classIndex)=1;
            bases{classIndex}(1:ambientDimension,1:dimensionClasses(classIndex),1)=normalizedSubspaces;
            vote{classIndex}(1)=1;
            sampleVotes(classIndex,sampleIndex) = 1;
        else
            % Compare with othre votes. If difference is big, create a new
            % basis
            newBasis = true;
            for basisIndex=1:basisCount(classIndex)
                angleDifference = subspace_angle(...
                bases{classIndex}(1:ambientDimension,1:dimensionClasses(classIndex),basisIndex),normalizedSubspaces);
                if angleDifference<angleTolerance
                    newBasis = false;
                    
                    % Update the existing basis by averaging.
                    oldBasis = sqrt(vote{classIndex}(basisIndex))*...
                    bases{classIndex}(1:ambientDimension,1:dimensionClasses(classIndex),basisIndex);
                    [UU,SS,VV]=svds([oldBasis normalizedSubspaces], dimensionClasses(classIndex));
                    bases{classIndex}(:,:,basisIndex) = UU(:,1:dimensionClasses(classIndex));
                    vote{classIndex}(basisIndex) = vote{classIndex}(basisIndex) + 1;
                    sampleVotes(classIndex, sampleIndex) = basisIndex;
                    break;
                end
            end
            if newBasis
                basisCount(classIndex) = basisCount(classIndex) + 1;
                bases{classIndex}(1:ambientDimension,1:dimensionClasses(classIndex),...
                    basisCount(classIndex))=normalizedSubspaces;
                vote{classIndex}(basisCount(classIndex)) = 1;
                sampleVotes(classIndex, sampleIndex) = basisCount(classIndex);
            end
        end
    end
end

% Merge subspace bases if they are closer than the angle tolerance
if MERGE_VOTES
    for classIndex=1:dimensionClassCount
        % Merge bases within each class
        index1=1;
        while index1<=basisCount(classIndex)-1
            index2=index1+1;
            while index2<=basisCount(classIndex)
                angleDifference = subspace_angle(...
                    bases{classIndex}(1:ambientDimension,1:dimensionClasses(classIndex),index1),...
                    bases{classIndex}(1:ambientDimension,1:dimensionClasses(classIndex),index2));
                if angleDifference<angleTolerance
                    % Merge two basis votes
                    basis_1=sqrt(vote{classIndex}(index1))*bases{classIndex}(1:ambientDimension,1:dimensionClasses(classIndex),index1);
                    basis_2=sqrt(vote{classIndex}(index2))*bases{classIndex}(1:ambientDimension,1:dimensionClasses(classIndex),index2);
                    [UU,SS,VV]=svds([basis_1 basis_2], dimensionClasses(classIndex));
                    bases{classIndex}(:,:,index1) = UU(:,1:dimensionClasses(classIndex));
                    vote{classIndex}(index1) = vote{classIndex}(index1) + vote{classIndex}(index2);
                    
                    % Change labels for all basis2 samples
                    sampleIndexes = (sampleVotes(classIndex,:)==index2);
                    sampleVotes(classIndex,sampleIndexes)=index1;
                    
                    % Move the voting array forward by one
                    for index3=index2+1:basisCount(classIndex)
                        bases{classIndex}(1:ambientDimension,1:dimensionClasses(classIndex), index3-1)=...
                            bases{classIndex}(1:ambientDimension,1:dimensionClasses(classIndex), index3);
                        vote{classIndex}(index3-1)=vote{classIndex}(index3);
                        sampleIndexes = (sampleVotes(classIndex,:)==index3);
                        sampleVotes(classIndex,sampleIndexes)=index3-1;
                    end
                    basisCount(classIndex)= basisCount(classIndex)-1;
                    vote{classIndex} = vote{classIndex}(1:end-1);
                    bases{classIndex} = bases{classIndex}(:,:,1:end-1);
                end
                index2 = index2 + 1;
            end
            index1 = index1+1;
        end
    end
end

% Detect highest votes within each class with the constraint that no 
% classes are empty.
subspaceCount = 0;
sampleLabels = zeros(1,sampleCount);

for classIndex=1:dimensionClassCount
    [ignored, peakVoteIndex]=sort(vote{classIndex},'descend');
    
    % Remove the empty classes
    peakVoteIndex=peakVoteIndex(1:find(peakVoteIndex>0,1,'last'));
    
    subspaceWithinClass = sum(subspaceDimensions==dimensionClasses(classIndex));
    if length(peakVoteIndex)<subspaceWithinClass
        warning('GPCA failed. The result contains empty classes. Try to decrease the angle tolerance parameter.');
        subspaceBases=[];
        sampleLabels = [];
        return;
    end
    
    indexPointer = 0;
    sampleVotesWithinClass = sampleVotes(classIndex,:);
    previousSubspaceCount = subspaceCount;
    for withinClassIndex=1:subspaceWithinClass
        
        indexPointer = indexPointer + 1;
        while indexPointer<=length(peakVoteIndex)
            samplePicks = (sampleVotesWithinClass==peakVoteIndex(indexPointer));
            
            % Have to test if the current subspace is too close to previous
            % ones in other subspace classes
            mergeSubspacesCrossClasses=0;
            for index=1:previousSubspaceCount
                angleDifference=subspace_angle(subspaceBases{index},bases{classIndex}(:,:,peakVoteIndex(indexPointer)));
                if angleDifference<angleTolerance
                    mergeSubspacesCrossClasses=index;
                    break;
                end
            end
            
            if mergeSubspacesCrossClasses==0
                % Add the new subspace into the record
                subspaceCount = subspaceCount + 1;
                subspaceBases{subspaceCount}=bases{classIndex}(:,:,peakVoteIndex(indexPointer));
                
                sampleLabels(samplePicks) = subspaceCount;
            else
                % Merge this subspace to another class
                sampleLabels(samplePicks) = mergeSubspacesCrossClasses;
            end
            
            % Withdraw the samples (samplePicks) from future voting counters.
            for sampleIndex=1:sampleCount
                if samplePicks(sampleIndex)==1
                    for index=classIndex+1:dimensionClassCount
                        vote{index}(sampleVotes(index,sampleIndex)) = vote{index}(sampleVotes(index,sampleIndex))-1;
                    end
                    
                end
            end
            sampleVotes(classIndex:dimensionClassCount,samplePicks)=0;
            
            % Loop Logic
            if mergeSubspacesCrossClasses==0
                break;
            else
                indexPointer = indexPointer + 1;
            end
        end
        if indexPointer>length(peakVoteIndex)
            warning('GPCA failed. The result contains empty classes. Try to decrease the angle tolerance parameter.');
            subspaceBases=[];
            sampleLabels = [];
            return;
        end
    end
end

% Convert dual space bases
for subspaceIndex=1:subspaceCount
    subspaceBases{subspaceIndex}=null(subspaceBases{subspaceIndex}');
end