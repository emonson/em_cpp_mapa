function X = rand_uniform_inside_hypersphere(ambientDimension, sampleCount)
% Generate sampleCount points sampled uniformly from within a unit hypersphere
% lying in an ambient space of dimension ambientDimension.
% Columns of X are data points.

% There are ways of doing this that don't require the computation of so
% many random numbers.  There is an article from the seventies on how
% to do this written in an era where this sort of thing took hours on
% contemporary hardware.

X = rand(ambientDimension, sampleCount)-0.5*ones(ambientDimension,sampleCount);
norms = sqrt(sum(X.*X,1));
radii = rand(1, sampleCount).^(1/ambientDimension);
X = X * diag(sparse(radii./norms));
