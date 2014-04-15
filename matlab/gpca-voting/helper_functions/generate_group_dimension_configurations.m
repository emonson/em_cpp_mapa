% Much faster way of computing all unique configurations of group
% dimensions.
function groupDimensionConfigurations = generate_group_dimension_configurations(ambientSpaceDimension, groupCount);

% groupCount = 3;
% ambientSpaceDimension = 4;

%veroneseSpaceDimension = nchoosek(groupCount+ambientSpaceDimension-1, groupCount);
%configurationCount = nchoosek(groupCount+ambientSpaceDimension - 2, groupCount);
%redundantConfigurationCount = (ambientSpaceDimension - 1)^groupCount;

% Generate the redundant arrays.
redundantArrays = cell(groupCount,1);
redundantArrays{1} = (1:ambientSpaceDimension - 1)';
for groupIndex = 2:groupCount,
    newColumn = reshape(repmat(1:(ambientSpaceDimension - 1), (ambientSpaceDimension - 1)^(groupIndex-1), 1), (ambientSpaceDimension - 1)^(groupIndex), 1);
    %    newColumn = reshape(repmat(1:(groupCount), (ambientSpaceDimension - 1)^(groupIndex-1), 1), (ambientSpaceDimension - 1)^(groupIndex), 1);
    redundantArrays{groupIndex} =  [newColumn repmat(redundantArrays{groupIndex - 1}, ambientSpaceDimension - 1, 1)];
end

redundantArray = redundantArrays{end};

% Pull out the values that are not increasing accross the row.
groupDimensionConfigurations = redundantArray(all(diff(redundantArray, 1, 2) >= 0, 2), :);
