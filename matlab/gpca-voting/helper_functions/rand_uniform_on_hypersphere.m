function X = rand_uniform_on_hypersphere(ambientDimension, sampleCount)
% Generate sampleCount points sampled uniformly from the surface of a unit hypersphere
% lying in an ambient space of dimension ambientDimension.
% Columns of X are data points.

% if ambientDimension == 1,
%     % For One Dimension, the sphere is just the two points {-1, 1}, thus we
%     % just sample uniformly in {-1, 1}
%     X = 2*rand(1,sampleCount)-1;
%     
% elseif ambientDimension == 2,
%     % For Two Dimensions, the sphere is just a circle in R2.
%     k = 1000;
%     theta = 2*pi*rand(1,sampleCount);
%     radii = rand(1, sampleCount).^.5;
%     X = [radii.*sin(theta); radii.*cos(theta)];
% 
% elseif ambientDimension > 2,

% Let the spherical symmetry of a vector of gaussians do most of the
% hard work for us.  Once we have a bunch of points projected onto a
% sphere, rescaling them by u^(1/ambientDimension) where u \in [0,1]
% will create a uniform density of points in the sphere.

% There are ways of doing this that don't require the computation of so
% many random numbers.  There is an article from the seventies on how
% to do this written in an era where this sort of thing took hours on
% contemporary hardware.

X = rand(ambientDimension, sampleCount)-0.5*ones(ambientDimension, sampleCount);
norms = sqrt(sum(X.*X,1));
X = X * diag(sparse(1./norms));
% end
