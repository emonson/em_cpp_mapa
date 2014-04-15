function positiveSample = find_good_sample(X, normals)

% Basic criteria for picking a good representive sample:
% 1. X coordinates far away from the origin
% 2. Its normal vector must not be short, which will be influenced more by
% the noise

X_PERSERVATION_RATIO = 0.5;
NORMAL_PERSERVATION_RATIO = 0.5;

[ambientDimension sampleNumber]= size(X);

for sampleIndex=1:sampleNumber
    xNorm(sampleIndex)=norm(X(:,sampleIndex));
end

[ignored, index] = sort(xNorm,'descend');

preserveIndex = ceil(sampleNumber*X_PERSERVATION_RATIO);
normalNorm=zeros(1, preserveIndex);
for sampleIndex = 1: preserveIndex
    normalNorm(sampleIndex)=norm(normals(:,index(sampleIndex)));
end

[ignored,normalIndex] = sort(normalNorm,'descend');

% Randomly pick one sample in 1/4 longest normal vector set.
positiveSample = index(normalIndex(ceil(preserveIndex*rand()*NORMAL_PERSERVATION_RATIO)));