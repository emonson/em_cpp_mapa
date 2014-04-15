function [p, flag] = modeling_success_rate(planeDims, dims)

dims = sort(dims); % true dimensions
K = length(dims);

n = length(planeDims);
flag = zeros(1,n);

cnt = 0;
for i = 1:n
    if length(planeDims{i}) == K &&  ~any(sort(planeDims{i}) - dims)
        cnt = cnt+1;
        flag(i) = 1;
    end
end

p = cnt/n;
