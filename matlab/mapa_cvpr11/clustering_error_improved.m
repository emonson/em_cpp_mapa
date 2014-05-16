function p = clustering_error_improved(clustered_labels, true_labels)

% Make sure both are row vectors (or columns will work)
clustered_labels = reshape(clustered_labels, 1, []);
true_labels = reshape(true_labels, 1, []);

Js = unique(clustered_labels);
Ks = unique(true_labels);
K = length(Ks);

planeSizes = zeros(1,K);
for k = 1:length(Ks)
    planeSizes(k) = sum(true_labels == Ks(k));
end

counts_mtx = zeros(K,K);
% k is the index of true labels
for k = 1:length(Ks)
    % j is the index of the inferred, clustering results labels
    for j = 1:length(Js)
        % count up how many matches there are with this k and j combination
        counts_mtx(k,j) = sum( (clustered_labels == Js(j)) & (true_labels == Ks(k)) );
    end
end

[n_correct, ~] = number_of_correctly_classified_points(counts_mtx);
p = 1 - n_correct/length(true_labels);

%%

function [n, opt_perm] = number_of_correctly_classified_points(counts_mtx)

K = size(counts_mtx,1);

% There are K! permutations of the integers 1:K
% We will consider k always to be the sequence 1:K
% and j will be the permuted forms

% Generate all permutations. Each row is one permutation.
permuted_js = perms(1:K);

% We'll keep track of the results for all of the permutations
% and then just use the permutation that gives us the best results
% (max number of correct answers)
n_permutations = size(permuted_js,1); % == K!
num_correct = zeros(1, n_permutations);

for pp = 1:n_permutations,
    for k = 1:K
        num_correct(pp) = num_correct(pp) + counts_mtx(k, permuted_js(pp,k));
    end
end

[n,I] = max(num_correct);
opt_perm = permuted_js(I);

