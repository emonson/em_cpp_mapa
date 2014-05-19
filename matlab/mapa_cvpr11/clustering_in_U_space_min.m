function indicesKmeans = clustering_in_U_space_min(U,opts)
% will cluster the rows of U (same number as the rows of A)
% minimal version using KMeansRex and only 'hard' seeds

K = size(U,2);

if opts.normalizeU,
    rowNorms = sqrt(sum(U.^2,2));
    rowNorms(rowNorms==0) = 1;
    U = U./repmat(rowNorms,1,K);
end

[~, indicesKmeans] = KMeansRex(U, K, 100, 'mapa');
% This is a C++ routine that has 0-based indices, so add one here
indicesKmeans = indicesKmeans + 1;

end



%figure; do_plot_data(U, indicesKmeans);
