function vStats = EstimateDimFromSpectra( cDeltas, S_MSVD, alpha0 )
%
% Estimates intrinsic dimensionality and good scales given the multiscale singular values
%
% IN:
%   cDeltas         : (#scales) vector of scales
%   S_MSVD          : (#scales)*(#dimensions) matrix of singular values: the (i,j) entry is the j-th singular value of a cell at scale cDeltas(i) around a point
%
% OUT:
%   vStats          : structure containing the following fields:
%                       DimEst      : estimate of intrinsic dimensionality
%                       GoodScales  : vector of indices (into the vector cDeltas) of good scales used for dimension estimate
%

[nScales,nDims] = size(S_MSVD);

s1 = cDeltas;

if nargin<3; alpha0 = 0.2; end

width = 5;
% iMax = 12;
iMax = min(12,nScales);
jMax = nScales;

vStats.GoodScales = [1,nScales];
vStats.DimEst = nDims;

%%
p = nDims;
sp = S_MSVD(:,p);

i = width; 
slope = compute_slope(s1(1:i), sp(1:i), 2);

while i<iMax && slope>0.1
    i=i+1; 
    slope = compute_slope(s1(i-width+1:i), sp(i-width+1:i), 2);
end

if i>iMax
    isaNoisySingularValue = false;
else
    j = i;
    while j<jMax && slope<=alpha0
        j = j+1; 
        slope = compute_slope(s1(j-width+1:j), sp(j-width+1:j), 2);
    end
    j = j-1;
   
    vStats.GoodScales = [i-1 j];   
    isaNoisySingularValue = true;
end

%%
while p>1 && isaNoisySingularValue
    
    p = p-1;
    sp = S_MSVD(:,p);
    
    % find a lower bound for the optimal scale
    i = width; slope = compute_slope(s1(1:i), sp(1:i), 2);
    while i<iMax && slope>0.1
        i=i+1; 
        slope = compute_slope(s1(i-width+1:i), sp(i-width+1:i), 2);
    end
    
    % find an upper bound for the optimal scale
    if slope<=alpha0
        
        j = i;
        while j<jMax && slope<=alpha0
            j = j+1; slope = compute_slope(s1(j-width+1:j), sp(j-width+1:j), 2);
        end
        j = j-1; 
        
        if mean((sp(j-width+1:j)-S_MSVD(j-width+1:j,p+1))./(S_MSVD(j-width+1:j,1)-S_MSVD(j-width+1:j,p+1)))> 0.2
            isaNoisySingularValue = false;
        else
            vStats.GoodScales = [i-1 j];
        end
                
    else
        
        isaNoisySingularValue = false;
        
    end
        
end

vStats.DimEst = p;

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function slope = compute_slope(s1, sp, method)

if nargin<3 || method==1,
    slope = (sp(end)-sp(1))/(s1(end)-s1(1));
else
    slope = (sum(s1.*sp) - sum(s1)*sum(sp)/numel(s1)) / (sum(s1.^2)-sum(s1)^2/numel(s1));
end

return;
