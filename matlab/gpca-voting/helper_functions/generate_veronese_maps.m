function [varargout] = generate_veronese_maps(data, highestOrder, status, memoryOptimization);
% mappedData = generate_veronese_maps(data, highestOrder);
%
%   Compute the Veronese map of a data set and, optionally, its
%   first and second derivatives (Gradient Vectors and Hessian Matrices)
%
%   For a set of N column vectors of dimension K specified in data,
%   generate_veronese_maps will compute the veronese maps of order [1 .. highestOrder],
%   along with the gradient and the Hessian (the matrix of all second order
%   partial derivatives) of the veronese maps.
%
%   The returned cell arrays have the following structure:
%
%   V{o}(i,n) : For the nth data vector, the ith entry of the oth order
%   Veronese map
%
%   D{o}(i,j,n) : For the nth data vector, the partial derivative with
%   respect to vector element j of the ith entry of the oth order Veronese
%   map
%
%   H{o}(i,j,k,n) : For the nth data vector, the second order partial
%   derivative with respect to the vector elements j and k of the ith entry
%   of the oth order Veronese map
%
% Example:
% Suppose we have the following point x in R3 or C3:
% [ x1     x2    x3     ]'
%
% The output of balls_and_bins(2,3) is:
%   [  2      0      0
%      1      1      0
%      1      0      1
%      0      2      0
%      0      1      1
%      0      0      2  ]
%
% Then the corresponding column of the second order Veronese map is:
% [ x1^2 * x2^0 * x3^0
%   x1^1 * x2^1 * x3^0
%   x1^1 * x2^0 * x3^1
%   x1^0 * x2^2 * x3^0
%   x1^0 * x2^1 * x3^1
%   x1^0 * x2^0 * x3^2 ]
%
% or:
%
% [x1^2 x1*x2 x1*x3 x2^2 x2*x3 x3^2]'
%
% The multiple linear subspace structure of the original data becomes a single
% higher dimensional linear subspace under this non-linear mapping.

% The default behavior is to return all of the Veronese Maps.
if nargin == 2,
    status = 'all';
end
if nargin<4
    memoryOptimization = false;
end

[K, N] = size(data);
V = cell(highestOrder, 1);
lowestOrder = 1;
if nargout ==1
    % Only output Veronese map
    if strcmp(status,'single')
        indices{highestOrder} = balls_and_bins(highestOrder, K);
        lowestOrder = highestOrder;
    else
        indices = balls_and_bins(highestOrder, K, 'all');
    end
else
    % Output Veronese map and its derivatives
    indices = balls_and_bins(highestOrder, K, 'all');
    D = cell(highestOrder, 1);
end
if nargout >= 3
    % Also output the Hessian matrices.
    H = cell(highestOrder, 1);
end

% One row. A non-zero value indicates a zero somewhere in the corresponding data vector.
zeroData = sum(data == 0);

% A non-zero value indicates that there is a zero somewhere in
% the data.
zeroFlag = sum(zeroData');

% A one indicates that there are no zeros in the corresponding data.
nonzeroData = ~zeroData;

% Compute the Natural Logarithm of the data (This is an elementwise
% operation).
logData = zeros(K, N);
warning('off');
logData = log(data);
warning('on');

% complexFlag is a signal to support veronese map of complex data.
complexFlag = ~isreal(data);

% Manage memory fragment.
if memoryOptimization
    pack;
end

for orderIndex = lowestOrder:highestOrder,
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Apply a trick to compute the Veronese map using the matrix
    % exponential and matrix multiplication.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    if zeroFlag == 0
        % No exact 0 element in the data, log(Data) is finite, with possible complex terms when the data value is negative
        if complexFlag
            V{orderIndex} = exp(indices{orderIndex} * logData);
        else
            V{orderIndex} = real(exp(indices{orderIndex} * logData));
        end
    else
        % If 0 values in the data, need to take care of the log(0) == -Inf terms
        [rows, ignored] = size(indices{orderIndex});
        if complexFlag
            V{orderIndex}(:,nonzeroData) = exp(indices{orderIndex}*logData(:,nonzeroData));
        else
            V{orderIndex}(:,nonzeroData) = real(exp(indices{orderIndex}*logData(:,nonzeroData)));
        end
        for dataPoint=1:N
            if zeroData(dataPoint)>0
                % data(dataPoint) has 0 elements that are left unprocessed above.
                for rowCount=1:rows
                    V{orderIndex}(rowCount,dataPoint) = prod(data(:,dataPoint)' .^ indices{orderIndex}(rowCount, :));
                end
            end
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the gradient and the Hessian of the veronese map.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if nargout >=2
        if (orderIndex == 1)
            D{orderIndex} = zeros(K,K,N);
            for d = 1:K
                D{orderIndex}(d,d,:) = ones(1,N);
            end % for d %

            if nargout >= 3,
                H{orderIndex} = zeros(K,K,K,N);
            end
        else
            clear D_o H_o
            Mn = size(indices{orderIndex},1);

            D_o = zeros(Mn,K,N);

            for d = 1:K
                % Take one column of the exponents array of order o
                D_indices = indices{orderIndex}(:,d);
                Vd = zeros(Mn, N);

                % Find all of the non-zero exponents, to avoid division by zero
                non_zeros = find(D_indices ~= 0);
                Vd(non_zeros,:) = V{orderIndex-1};

                % Multiply the lower order veronese map by the exponents of the
                % relevant vector element

                %            D_o(:,d,:) = repmat(D_indices, 1, N) .* Vd;
                temp = repmat(D_indices, 1, N) .* Vd;
                %            D_o(:,d,:) = permute(temp, [1 3 2]);  % Recoded for GNU Octave compatibility.
                D_o(:,d,:) = temp;

                if (nargout >= 3) && ((strcmp(status,'all')) || (orderIndex==highestOrder))
                    for h = d:K
                        H_indices = indices{orderIndex}(:,h);
                        Vh = zeros(Mn, N);
                        non_zeros = find(H_indices ~= 0);
                        Vh(non_zeros, :) = D{orderIndex-1}(:,d,:);
                        H_o(:,d,h,:) = repmat(H_indices, 1, N) .* Vh;
                        if (d ~= h)
                            H_o(:,h,d, :) = H_o(:,d,h,:);
                        end % if %
                    end % for h %
                end % if %
            end % for d %

            if (nargout >= 3) && ((strcmp(status,'all')) || (orderIndex==highestOrder))
                H{orderIndex} = H_o;
            end

            D{orderIndex} = D_o;
            if strcmp(status,'single')
                % Clear previous result to save space.
                D{orderIndex-1} = [];
                if memoryOptimization
                    pack;
                end
            end
        end % if %
    end % if %
end % for orderIndex %

if strcmp(status,'all')
    varargout{1} = V;
    if nargout >= 2, varargout{2} = D; end
    if nargout >= 3, varargout{3} = H; end
else
    varargout{1} = V{highestOrder};
    if nargout >= 2, varargout{2} = D{highestOrder}; end
    if nargout >= 3, varargout{3} = H{highestOrder}; end
end

