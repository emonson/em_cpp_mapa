% Test function for hypersphere area.

clc; clear all; close all;

% Check the area for a couple special cases.
r = 13;
hypersphere_area(1, r) - 2
hypersphere_area(2, r) - 2*pi*r
hypersphere_area(3, r) - 4 * pi * r^2

% Display the volume of the unit hypersphere as a function of dimension.
dimensionCount = 20;
area = zeros(1,dimensionCount);
for dimensionIndex = 1:dimensionCount,
    area(dimensionIndex) = hypersphere_area(dimensionIndex);
end
plot(1:dimensionCount, area, '*')
title('Area of a unit hypersphere a function of Dimension')
xlabel('Dimension')
ylabel('Area')
