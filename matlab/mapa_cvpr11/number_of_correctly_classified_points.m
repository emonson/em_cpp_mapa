function [max_correct, opt_perm] = number_of_correctly_classified_points(counts_mtx)

K = size(counts_mtx,1);
n_perms = factorial(K);

% There are K! permutations of the integers 1:K
% We will consider k always to be the sequence 1:K
% and j will be the permuted forms

% Since there are K! permutations, we won't store them all, but generate
% them on the fly and just keep the best one.
init_perm = (1:K)';
curr_perm = init_perm;
max_correct = -inf;

for ii = 1:n_perms
    num_correct = 0;
    for k = 1:K
        num_correct = num_correct + counts_mtx(k, curr_perm(k));
    end
    if num_correct > max_correct,
        max_correct = num_correct;
        opt_perm = curr_perm;
    end
    curr_perm = nextperms(curr_perm, 1);
end