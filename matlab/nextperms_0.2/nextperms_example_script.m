% Example use of NEXTPERMS
%
% NEXTPERMS is setup with some behaviors that are a little tricky to 
% handle.  In particular it does not check for wraparound if you request
% more permutations that a complete cycle.  And it does not include
% the input as the first column of output; output starts at the
% permutation after the input.
%
% This example demonstrates how to handle these behaviors for a typical
% use case of running through one complete cycle of permutations
%

v = uint8(1:13);        % Vector to permute
maxblocksize = 1e7;     % Block size of permutations to generate per batch

len = length(v);
nperms = factorial(len);
blocksize = min(nperms, maxblocksize); % Check that we didn't set blocks larger than full set of perms
nblocks = floor(nperms / blocksize);

% We will need a last block if nperms didn't divide evenly by blocksize
lastblock = nblocks*blocksize < nperms;

% Initialize with last perm so that first block starts with first
% lexicographic perm
currperm = sort(v, 'descend');

for i = 1:nblocks
    t = tic;
    block = nextperms(currperm, blocksize);
    currperm = block(:,end);
    
    % Process blocks here
    fprintf(1, 'Processed block %d/%d in %f s\n', ...
        i, nblocks+lastblock, toc(t))
end

% Last block is shorter
if lastblock
    block = nextperms(currperm, nperms - (blocksize*nblocks));    
    % Process lastblock here
    fprintf(1, 'Processed block %d/%d in %f s\n', ...
        nblocks+1, nblocks+1, toc(t))
end 