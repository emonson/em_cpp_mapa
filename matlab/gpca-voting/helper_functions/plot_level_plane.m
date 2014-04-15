% Add the zero plane to a three dimensional plot.
function plot_level_plane(zValue)

GRIDCOUNT = 2; % The number of gridlines.

tempLimits = axis;
[X,Y] = meshgrid(linspace(tempLimits(1),tempLimits(2), GRIDCOUNT),...
    linspace(tempLimits(3), tempLimits(4), GRIDCOUNT));
Z = zeros(size(X));
Z(:,:,:) = zValue;

pl = surf(X,Y,Z);
set(pl,'FaceColor','black','FaceAlpha',.2)

