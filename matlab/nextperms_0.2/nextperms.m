% NEXTPERMS     Generate a block of lexicographic permutations
% usage: perms = nextperms(v, n)
%
% This is documentation for the nextperms.cpp Mex file.  To use this
% function you must first compile it for your system with:
%   'mex nextperms.cpp' % from the directory holding nextperms.cpp
%
% This is a simple wrap of the C++ STL next_permutation function.  This
% generates lexicographic permutations.  The method is not defined in 
% the spec, but presumably the algorithm is as given here:
%   http://en.wikipedia.org/wiki/Permutation#Generation_in_lexicographic_order
%
% For example, nextperms(1:3, 4) will generate the 4 permutations after
% 1 2 3.  Note that the output is column-wise regardless of the shape
% of the input:
%
%   nextperms(1:3, 4)
%
%       ans =
%
%            1     2     2     3
%            3     1     3     1
%            2     3     1     2
%
% Note that the input permutation is not included in the returned
% output.
%
% *** IMPORTANT ***
% This does not have any protection against wrapping around back to the
% beginning of the permutations, so if n > factorial(numel(v)) you will
% wrap around and start repeating the earlier permutations.  It is up
% to the user to only request the appropriate number of permutations.
%
% Typical use case would be if you need to iterate through a large 
% number of permutations that will not all fit in memory at once.
%
% (If your number of permutations is not large, you could just use 
% Matlab's PERMS function.  Note that this does not generate the
% permutations in lexicographic order.  On my system, NEXTPERMS is also 
% significantly faster.)
%
% To get all permutations in lexicographic order, you could take
% advantage of the wrap around property by inputing the reverse V,
% i.e. the last lexicographic permutation:
%   nextperms(4:-1:1, factorial(4)); % All perms in lexicographic order
% 
% See NEXTPERMS_EXAMPLE_SCRIPT for a typical use case to run through
% all permutations in batches.
%
% Any numeric type can be used as input V.
%


% Version 0.2
% Peter H. Li 21-DEC-2013
% As required by MatLab Central FileExchange, licensed under the FreeBSD License
