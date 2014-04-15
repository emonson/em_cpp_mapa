function X = rand_hypersphere(ambientDimension, sampleCount)
% Generate sampleCount points sampled uniformly from within a hypersphere
% of a random radius lying in an ambient space of dimension ambientDimension.
% Columns of X are data points.

X = 100*rand()*(rand(ambientDimension, sampleCount)-0.5*ones(ambientDimension,sampleCount));
