% hypersphere_volume(ambientDimension, R)
%
% Compute the volume of a generalized sphere (i.e. including dimensions
% greater than three.
%
% If the second input is not specified, the value of the radius is taken to
% be one.
%
% Implements a formula found at
% http://planetmath.org/encyclopedia/Sphere.html
%
% V(n) = \frac{\pi^{\frac{n}{2}}r^n}{\Gamma(\frac{n}{2}+1)}
function volume = hypersphere_volume(ambientDimension, r)

if nargin == 1,
    r = 1;
end

% Equations below have already been adjusted
%n = ambientDimension - 1;
n = ambientDimension;

if ambientDimension == 1,
    volume = 2*r; % The volume of a sphere in R1 is just the length from -R to R.
    return
elseif ambientDimension == 2,
    volume = pi*r^2;  % For faster computation, use the familiar formula.
    return
elseif ambientDimension == 3,
    volume = 4/3 * pi * r^3;  % For faster computation, use the familiar formula.
    return
elseif ambientDimension > 3,
    % The following formula is only good for n >= 1.
    numerator = pi^(n/2) * r^n;  % Notice that this includes a term that is just the volume of the bounding hypersphere.
    denominator = gamma(n/2 + 1); % Gamma would probably be a bitch to compute; glad MATLAB provides it.
    volume = numerator/denominator;
    return
else
    error('The ambient dimension must be a natural number');
end

