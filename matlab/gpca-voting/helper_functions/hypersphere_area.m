% hypersphere_area(ambientDimension, R)
%
% Compute the area of a generalized sphere (i.e. including dimensions
% greater than three.
%
% If the second input is not specified, the value of the radius is taken to
% be one.
%
% Implements a formula found at
% http://planetmath.org/encyclopedia/Sphere.html
%
% A(n) = \frac{2\pi^{\frac{n+1}{2}}}{\Gamma\left(\frac{n+1}{2}\right)}
% A(n) = \frac{2\pi^{\frac{n+1}{2}}}{\Gamma\left(\frac{n+1}{2}\right)} * R^n
function area = hypersphere_area(ambientSpaceDimension, r)

if nargin == 1,
    r = 1;
end    
if nargin == 0,
    ambientDimension = 4;
end

n = ambientSpaceDimension - 1;

if ambientSpaceDimension == 0,
    area = 1; % The area of a one dimensional sphere just counts the points on the surface.
    warning('Not really sure what a area of sphere in zero dimensions is. Assuming one.')
    return
elseif ambientSpaceDimension == 1
    area = 2;  % Just counting the two points at the ends of the line segment.
elseif ambientSpaceDimension == 2,
    area = 2*pi*r;  % For faster computation, use the familiar formula.
    return
elseif ambientSpaceDimension == 3,
    area = 4 * pi * r^2;  % For faster computation, use the familiar formula.
    return
elseif ambientSpaceDimension > 3,
    % The following formula is only good for n >= 1.
    numerator = 2 * pi^((n+1)/2) * r^n;  % Notice that this includes a term that is just the volume of the bounding hypersphere.
    denominator = gamma((n+1)/2); % Gamma would probably be a bitch to compute; glad MATLAB provides it.
    area = numerator/denominator;
    return
else
    error('The ambient dimension must be a natural number');
end
