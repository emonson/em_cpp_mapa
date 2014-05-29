function [p, counts_mtx, opt_perm] = clustering_error_improved(clustered_labels, true_labels)

% Right now this only works for equal numbers of clusters in both the true
%   and inferred labels. 
% For unequal numbers, the inefficient way would be to make K equal to the
%   larger of the two numbers of labels and do all the rest the same. That
%   is what we have implemented at this point. 
% The most efficient solution would be to make J the larger of the two and
%   K the smaller, and then instead of all of the permutations in the
%   subroutine, generate all of the nchoosek combinations, (j choose k), and
%   then generate all of the permutations of each of those and check them for
%   which is best. That would make k! * (n!/((n-k)!*k!)) possibilities over
%   which to find the max.

% Make sure both are row vectors (or columns will work)
clustered_labels = reshape(clustered_labels, 1, []);
true_labels = reshape(true_labels, 1, []);

Js = unique(clustered_labels);
Ks = unique(true_labels);
K = max([length(Js) length(Ks)]);

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

if K > 12,
    fprintf(1,'Too many clusters to calculate error in the way we currently implemented\n');
    n_correct = NaN;
    opt_perm = ones(K,1);
else
    [n_correct, opt_perm] = number_of_correctly_classified_points(counts_mtx);
end
p = 1 - n_correct/length(true_labels);
