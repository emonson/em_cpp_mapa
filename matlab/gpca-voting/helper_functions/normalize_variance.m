function [X,H] = normalize_variance(x, H)
% image_coordinate_cormalization balaces the magnitute difference on each
% entry of x. H returns the mean and the standard deviation of the entries.

[dimension, number] = size(x);

if nargin==1
    % Normalize the variation 
    for k=1:dimension
        H(k) = std(x(k,:));
        if H(k)>0
            X(k,:) = x(k,:)/H(k);
        else
            X(k,:) = x(k,:);
        end
    end
else
    % Denormalize the variation
    for k=1:dimension
        if H(k)>0
            X(k,:) = x(k,:)*H(k);
        else
            X(k,:) = x(k,:);
        end
    end
end