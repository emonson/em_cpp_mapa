% Test function for hypersphere volume.

clc; clear all; close all;

% Check the volume for a couple special cases.
r = 13;
hypersphere_volume(1, r) - 2*r
hypersphere_volume(2, r) - pi*r^2
hypersphere_volume(3, r) - 4/3 * pi * r^3

% Display the volume of the unit hypersphere as a function of dimension.
dimensionCount = 20;
volume = zeros(1,dimensionCount);
for dimensionIndex = 1:dimensionCount,
    volume(dimensionIndex) = hypersphere_volume(dimensionIndex);
end
plot(1:dimensionCount, volume, '*')
title('Volume of a unit hypersphere a function of Dimension')
xlabel('Dimension')
ylabel('Volume')
