% Generates a single entry in the veronese map codimension lookup table.  This can be run by hand
% for the slower cases, or called by other functions to fill out the table
% quickly.
%
function generate_veronese_rank_lookup_table_entry(ambientSpaceDimension, groupCount);

RATIO_THRESHOLD = 1e-10;
USE_RAO_CONJECTURE = false;
memoryOptimization = false;
% Generate a table of values that represent unique group
% dimension (or codimension) configurations.
basisCodimensionsArray = generate_group_dimension_configurations(ambientSpaceDimension, groupCount);
configurationCount = size(basisCodimensionsArray,1);
veroneseCodimensionsVector = zeros(configurationCount, 1);

% Load previous results
load rankTable.mat;
if (size(rankTable,1)<ambientSpaceDimension) || (size(rankTable,2)< groupCount) || isempty(rankTable{ambientSpaceDimension,groupCount-1,1})
    RECURSIVE_RESULT=0;
else
    RECURSIVE_RESULT=1;
    recursiveTable = rankTable{ambientSpaceDimension,groupCount-1,2};
end

cacheFileName=strcat('table',int2str(ambientSpaceDimension),int2str(groupCount),'.mat');
cacheFileID=fopen(cacheFileName,'r');
if cacheFileID ~= -1
    % Reload previous results
    load(cacheFileName);
    startingPoint=length(veroneseCodimensionsVector)+1;
else
    startingPoint = 1;
end
clear cacheFileName cacheFileID 
if memoryOptimization
    pack;
end

for configurationIndex = startingPoint:configurationCount,
    currentConfiguration = basisCodimensionsArray(configurationIndex, :);

    if (currentConfiguration(end-1)==1)
        % Trival case.
        veroneseCodimensionsVector(configurationIndex)=basisCodimensionsArray(configurationIndex,end);
    elseif RECURSIVE_RESULT && (currentConfiguration(1)==1)
        veroneseCodimensionsVector(configurationIndex)=recursiveTable(configurationIndex);
    elseif (USE_RAO_CONJECTURE && is_rao_conjecture_elligible(basisCodimensionsArray(configurationIndex-1, :), currentConfiguration, ambientSpaceDimension))
        veroneseCodimensionsVector(configurationIndex) = veroneseCodimensionsVector(configurationIndex-1) + sum(currentConfiguration > 1);
    else
        % Perform several repetitions to be more confident we got
        % the answer correct.
        % Generate the data set.
        X = generate_samples_fast(ambientSpaceDimension,  ambientSpaceDimension - basisCodimensionsArray(configurationIndex, :));

        % Compute the veronese map of the data
        if memoryOptimization
            pack;
        end
        V = generate_veronese_maps(X, groupCount, 'single', memoryOptimization);
        clear X
        
        S = svd(V);  % Computing just the singular values.  Note that calculating the vectors takes much longer.
        % Look for the first big rank drop
        ratio = S ./ cumsum(S);
        veroneseRankGuess = min(find(ratio < RATIO_THRESHOLD)) - 1;

        % Save the value and continue.
        veroneseCodimensionsVector(configurationIndex) = size(V,1) - veroneseRankGuess;
        clear V S
        if memoryOptimization
            pack;
        end
    end
    disp(['[' num2str(basisCodimensionsArray(configurationIndex, :)) '] ---> ' num2str(veroneseCodimensionsVector(configurationIndex)) '  Percent Finished: ' num2str(floor(100*configurationIndex / size(basisCodimensionsArray,1)))]);
    if (configurationIndex | 10) == 0
        save(cacheFileName,'veroneseCodimensionsVector');
    end
end %for configurations

rankTable{ambientSpaceDimension,groupCount,1}=veroneseCodimensionVector;
rankTable{ambientSpaceDimension,groupCount,2}=basisCodimensionsArray;
save rankTable.m rankTable