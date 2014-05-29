function [planeDims, labels, err, planeCenters, planeBases] =  spectral_analysis(X, U, allPtsInOptRegions, invColMap, localDims, nOutliers)

K = size(U,2);

SCCopts = struct();
SCCopts.normalizeU = 1;
SCCopts.seedType = 'hard';
indicesKmeans = clustering_in_U_space_min(U,SCCopts);

planeDims = zeros(1,K);
for k = 1:K
    % Find the original point indices of the rows of A/U in this cluster
    class_k = allPtsInOptRegions(indicesKmeans == k);
    % Figure out which of these points are seed points
    temp = invColMap(class_k);  
    temp = temp(temp>0);
    % Then see what dimensionality most of these seed points in this
    % cluster have
    planeDims(k) = mode(localDims(temp));
end

[planeCenters, planeBases] = computing_bases(X(allPtsInOptRegions,:), indicesKmeans, planeDims);
dists = p2pdist(X,planeCenters,planeBases);

%[N,D] = size(X);
%dists = dists./repmat(D-planeDims, N, 1);

[dists,labels] = min(dists,[],2);

if nOutliers>0
    % new labels
    labels1 = labels;
    objectiveFun1 = sum(sqrt(dists));
    outliers1 = [];
    % old labels
    objectiveFun = Inf;
    labels = [];
    outliers = [];
    % NOTE: I don't quite see how this is a progressive optimization...?
    while objectiveFun1<objectiveFun
        % if we're doing better, go ahead and grab the labels from the last round
        labels = labels1;
        % and keep track of sum of distances
        objectiveFun = objectiveFun1;
        % and lock in the outliers from last time, too
        outliers = outliers1;
        % sort descending so first points are furtheset pointss
        [~, I] = sort(dists, 'descend');
        % grab indices of points farthest from any planes
        outliers1=I(1:nOutliers);
        % setting labels as zero gets them ignored in computing_bases
        labels1(outliers1)=0;
        % compute new planes ignoring these outlying points
        [planeCenters, planeBases] = computing_bases(X, labels1, planeDims);
        % calculate distances from all points to these new bases
        dists = p2pdist(X,planeCenters,planeBases);
        % figure out which groups all points should belong to using these new bases
        [dists,labels1] = min(dists,[],2);
        objectiveFun1 = sum(sqrt(dists));
    end
    labels(outliers)=0;
end

err = L2error(X, planeDims, labels);
