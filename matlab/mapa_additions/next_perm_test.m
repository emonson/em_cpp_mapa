K = 7;
n_perms = factorial(K);

% There are K! permutations of the integers 1:K
% We will consider k always to be the sequence 1:K
% and j will be the permuted forms

% Since there are K! permutations, we won't store them all, but generate
% them on the fly and just keep the best one.
init_perm = [1:K]';
next_perm = 1:K;
max_correct = -inf;
ii = 0;

while true,
    if mod(ii, 1000) == 0,
        disp([ii n_perms]);
    end
    curr_perm = next_perm;
    num_correct = 0;
    % next_perm = nextperm(curr_perm, K);
    next_perm = nextperms(curr_perm, 1);
    % check if done with all permutations
    if next_perm == init_perm,
        break;
    end
    ii = ii + 1;
end

