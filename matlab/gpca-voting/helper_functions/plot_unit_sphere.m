% plot_unit_sphere
% Plot a wiremesh of the unit sphere onto the current axes.
function out = plot_sphere(varargin);

if nargin > 0,
    pointCount = varargin{1};
else, 
    pointCount = 20;
end

[sphereX, sphereY, sphereZ] = sphere(pointCount);
mesh(sphereX, sphereY, sphereZ, 1*ones(pointCount,pointCount));
hidden off;
axis equal;
colormap(colorcube)
