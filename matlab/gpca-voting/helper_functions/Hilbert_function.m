function result = Hilbert_function(ambientDimension, subspaceDimensions);
% This function calculates the Hilbert function value.
% References:
%   Harm Derksen. Hilbert series of subspace arrangements.

subspaceCount = length(subspaceDimensions);
subspaceCodimensions=ambientDimension*ones(1,subspaceCount)-subspaceDimensions;
if sum(subspaceDimensions<0)>0 || sum(subspaceDimensions>=ambientDimension)
    error('Subspace dimensions are out of range.');
end

% Generate index set S of all proper subsets of subspaceCodimensions.
S={};
for index=1:subspaceCount
    new_S = generate_subsets(subspaceCount, index);
    for lengthIndex=1:size(new_S,1)
        S{end+1}=new_S(lengthIndex,:);
    end
end

numeratorFactor = subspaceCount+ambientDimension-1;
denominatorFactor = ambientDimension - 1;
result = nchoosek(numeratorFactor, denominatorFactor); % when S is empty set
for index=1:length(S)
    C_S=sum(subspaceCodimensions(S{index}));

    if C_S<ambientDimension
        result = result + (-1)^length(S{index})*nchoosek(numeratorFactor-C_S, denominatorFactor - C_S);
    end
end