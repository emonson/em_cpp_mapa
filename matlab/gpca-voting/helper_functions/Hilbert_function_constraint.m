function result = Hilbert_function_constraint(ambientDimension, subspaceDimensions);

load rankTable.mat
subspaceCount = length(subspaceDimensions);
subspaceCodimensions=ambientDimension*ones(1,subspaceCount)-subspaceDimensions;
subspaceCodimensions=sort(subspaceCodimensions);

% Remove Hyperplances
while (subspaceCount>2) && (subspaceCodimensions(1)==1)
    subspaceCodimensions=subspaceCodimensions(2:end);
    subspaceCount = subspaceCount - 1;
    if subspaceCount==0
        % All hyperplanes
        result = 1;
        return;
    end
end

codimensionCombinations=rankTable{ambientDimension,subspaceCount,1};
charDimensions = rankTable{ambientDimension,subspaceCount,2};

% Find the right entry
entryFound = false;
entryIndex = 0;
while (entryFound==false) && (entryIndex<size(codimensionCombinations,1))
    entryIndex = entryIndex + 1;
    if issame(codimensionCombinations(entryIndex,:),subspaceCodimensions)
        entryFound = true;
    end
end

if entryFound == false
    error('The dimension combination is not in the Hilbert function constraint table.');
end

result = charDimensions(entryIndex);