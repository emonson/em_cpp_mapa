function subsetIndices = generate_subsets(setSize, subsetSize)
% This function generate a complete list of subset indices of setSize
% elements in a set. The subset size is given by the parameter subsetSize

more = 1;

m2 = 0;
m = subsetSize;
subsetIndices = [];
while (1)
    for j = 1 : m
        a(subsetSize+j-m) = m2 + j;
    end

    subsetIndices(end+1,:) = a;
    more = ( a(1) ~= (setSize-subsetSize+1) );
    
    if more==0
        break;
    end
    
    if ( m2 < setSize-m )
        m = 0;
    end
    m = m + 1;
    m2 = a(subsetSize+1-m);
end