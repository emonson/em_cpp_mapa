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

scale_distances = cDeltas;

if nargin<3; alpha0 = 0.2; end

width = 5;
% iMax = 12;
iMax = min(12,nScales);
jMax = nScales;

vStats.GoodScales = [1,nScales];
vStats.DimEst = nDims;

%%

% Start with the smallest singular values, which should be the most noisy
p = nDims;
spectrum = S_MSVD(:,p);

i = width; 
slope = compute_slope(scale_distances, spectrum, width, i);

% Test whether slope is already > 0.1
% If so, increment r/delta index, computing slope and testing slope along
% the way. This is to get past any initial rise, and find at what index, if
% any, the slope flattens out below 0.1
while i<=iMax && slope>0.1
    i=i+1; 
    slope = compute_slope(scale_distances, spectrum, width, i);
end

% Testing whether slope never flattened out. If it didn't
% flatten out (go < 0.1), then it's "not a noisy singular value" and don't
% need to check when the flat started to rise again in the else section.
% This would also mean we don't have to test any lower dim. All scales are
% "good", and dimensionality of the manifold is the full dimensionality of
% the system.
if (i > iMax),
    % Never flattened out, so always rising and thus not "noisy", so set
    % the flag to keep it from going through any more tests
    isaNoisySingularValue = false;
else
    % flattened out at some point, so set j = the index at which the slope
    % dropped below 0.1
    j = i;
    % find r/delta index at which slope rises again to go above alpha0, 
    % or we hit the highest scale
    while j<jMax && slope<=alpha0
        j = j+1; 
        slope = compute_slope(scale_distances, spectrum, width, j);
    end
    % go to index before end of data or slope went above alpha0
    j = j-1;
    
    % set min at one before slope flattened out, and max one before rose
    % above alpha0
    vStats.GoodScales = [i-1 j];   
    % I think this just gets this flag ready for the next test, which will
    % break out if in decreasing the singular value index (dim), we hit a
    % case where curves stop being "noisy" and never flatten out.
    isaNoisySingularValue = true;
end

%%

% now loop through decreasing dim index down to 1, and break out early if
% we hit a dim at which the slope never flattens out below 0.1 at any
% r/delta
while p>1 && isaNoisySingularValue
    
    p = p-1;
    spectrum = S_MSVD(:,p);
    
    % find a lower bound for the optimal scale
    % try to find an r/delta index at which the slope drops below 0.1
    % (flattens out)
    i = width; 
    slope = compute_slope(scale_distances, spectrum, width, i);
    while i<iMax && slope>0.1
        i=i+1; 
        slope = compute_slope(scale_distances, spectrum, width, i);
    end
    
    % find an upper bound for the optimal scale
    % If the slope never even dropped below alpha0 (note, different test
    % than above...), then don't bother trying to find another rise from
    % the nonexistent flat portion, and break out of this loop because it's
    % assumed once the slope always stays above alpha0, none of the rest of
    % the dims are "noisy", i.e. those larger singular value dims are all
    % part of the manifold around this net point, so we can set the
    % estimation of the dimension, and we won't get any more help from
    % these dims judging the window of good scales (r/delta indices)
    if slope<=alpha0
        
        % Set j either at iMax or at the r/delta index at which the slope
        % flattened out below 0.1
        j = i;
        % find the point at which the slope again rises, now above alpha0,
        % or when we hit the end of the data
        while j<jMax && slope<=alpha0
            j = j+1; 
            slope = compute_slope(scale_distances, spectrum, width, j);
        end
        % set j back to the index before which the slope rose above alpha0,
        % or one before the end of the data
        j = j-1; 
        
        % If the curve flattens out for a while, but the gap between it and
        % the previous dim spectrum is wide enough, going to count it as a
        % "non-noisy" real dim that's part of the manifold
        current_to_prev_spectra_diff = (spectrum(j-width+1:j)-S_MSVD(j-width+1:j,p+1));
        largest_to_prev_spectra_diff = (S_MSVD(j-width+1:j,1)-S_MSVD(j-width+1:j,p+1));
        mean_fractional_spectrum_rise = mean(current_to_prev_spectra_diff./largest_to_prev_spectra_diff);
        if mean_fractional_spectrum_rise > 0.2
            % real manifold dim, so set flag to break out of dim decrease
            % loop
            isaNoisySingularValue = false;
        else
            % still a noisy scale with a low-enough flat portion that is
            % helping us find the "good scales", so update those. 
            % We want to use the values from the last dim found to be
            % "noisy", i.e. flat in some portion and not risen enough from
            % the previous, higher, dim, so keep updating GoodScales each
            % time, so when we break out of the loop with a noisy == false
            % will have the proper r/delta indices recoreded
            vStats.GoodScales = [i-1 j];
        end
                
    else
        % slope never dropped below alpha0, so set flag to break out of
        % loop
        isaNoisySingularValue = false;
        
    end
        
end

vStats.DimEst = p;

return;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function slope = compute_slope(distances, spectrum, width, idx)
    s1 = distances(idx-width+1:idx);
    sp = spectrum(idx-width+1:idx);
    slope = (sum(s1.*sp) - sum(s1)*sum(sp)/numel(s1)) / (sum(s1.^2)-sum(s1)^2/numel(s1));
end

