function X=generate_image_noise(X, noiseLevel)

% Generate Gaussian noise
noise = noiseLevel*(rand(size(X))-0.5*ones(size(X)));

X=X+noise;
