function pointToSpaceDistance = point_to_space_distance(samples, subspaceBasis)
% function pointToSpaceDistance = point_to_space_distance(samples, subspaceBasis)
%
% Computes the Euclidean distance between one or more vectors and a subspace. 
%  (i.e. the normal of (the vectors minus their projection onto the subspace ))
% 
% Inputs:
%   samples -       The samples to be projected, represented as columns.
%
%   subspaceBasis - The basis for the subspace to project onto; each column
%                   is a basis vector.
%
% Output:
%   pointToSpaceDistance -  A row vector of distances between the points and
%                           the subspace.

if (size(samples,2) == 1),
    pointToSpaceDistance = norm(samples - subspaceBasis*(subspaceBasis'*samples));
else
    perpVectors = samples - subspaceBasis*(subspaceBasis'*samples);
    pointToSpaceDistance = sqrt(sum(perpVectors.*perpVectors));
end

