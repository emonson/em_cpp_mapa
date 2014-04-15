function p = clustering_error(indices,trueLabels)

% K = length(planeSizes);
% count = 0;
% inds = 1:K;
% for k = 1:K
%     num = zeros(length(inds),1);
%     for j = inds
%         num(j) = sum((indices(sum(planeSizes(1:k-1))+1:sum(planeSizes(1:k)))==j));
%     end
%     [max_num, I] = max(num);
%     inds = setdiff(inds, I(1));
%     count = count+max_num;
% end
% p = 1-count/sum(planeSizes);

[sortedLabels, inds_sort] = sort(trueLabels, 'ascend');
indices = indices(inds_sort);

N = length(trueLabels);
K = sortedLabels(N);

planeSizes = zeros(K,1);
k = 1;
i = 1;
ini = 0;
while k<=K 
    while i<=N && sortedLabels(i)== k
        i = i+1;
    end
    planeSizes(k)=i-1-ini;
    ini = i-1;
    k=k+1;
end

num = zeros(K,K);
for k = 1:K
    for j = 1:K
        num(k,j) = sum((indices(sum(planeSizes(1:k-1))+1:sum(planeSizes(1:k)))==j));
    end
end

p = 1-number_of_correctly_classified_points(num)/sum(planeSizes);

%%

function n = number_of_correctly_classified_points(num)

K = size(num,1);

if K>2
    n = zeros(K,1);
    for j = 1:K
        n(j) = num(1,j)+number_of_correctly_classified_points(num(2:end,[1:j-1 j+1:K]));
    end
    n = max(n);
elseif  K == 2 
    n = max(num(1,1)+num(2,2), num(1,2)+num(2,1));
else
    n = num;
end