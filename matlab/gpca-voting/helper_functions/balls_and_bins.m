function outputCellArray = balls_and_bins(varargin)
% Enumerates all possible groupings of identical objects, i.e. balls in bins.
% Note: This happens to be the same problem as computing the exponenets of the veronese map.
%
% ARRANGEMENTS = BALLS_AND_BINS(BALLCOUNT,BINCOUNT) enumerates all of the 
% possible arrangements of BALLCOUNT identical balls in BINCOUNT bins.
% Each column of ARRANGEMENTS corresponds to a bin.
% Each row of ARRANGEMENTS is a possible arrangement of the balls, 
% with 0,1,2,3,... coresponding to 0,1,2,3,... balls in the bin.
% The arrangements are given in reverse Lexicographic order.
%
% ARRANGEMENTS = BALLS_AND_BINS(MINBALLS, MAXBALLS, MINBINS, MAXBINS) 
% Returns a cell array ARRANGEMENTS of solutions for the specified range
% of balls and bins:
% Each cell array column corresponds to a number of balls.
% Each cell array row corresponds to a number of bins.

% Some Input Error Checking
if (nargin ~= 2 && nargin  ~=3),
    error('Sorry, the function ''balls_and_bins'' must be called with either 2 or 3 parameters.');
end

% Unpack the inputs
ballCount = varargin{1};
binCount = varargin{2};
   
% if nargin == 4,
%     ballCount = varargin{2};
%     binCount = varargin{4};
% end

% Some more input checking
if (ballCount < 0 || binCount < 0 || mod(ballCount,1)~=0 || mod(binCount,1)~=0), 
    error('Sorry, the parameters of the function ''balls_and_bins'' must be positive integers.');
end

% Create a two dimensional cell array to hold solutions for smaller values of both ballCount and binCount.
outputCellArray2D = cell(ballCount,binCount);

% The top row of the cell array is given by identity matrices.
for column = 1:binCount,  % Column of the cell array, that is.
    outputCellArray2D{1,column} = eye(column);
end

% The leftmost column of cell array (1 Bin) is also trivial.
for row = 1:ballCount,
    outputCellArray2D{row,1} = row;
end

% Apply method of computing that exploits a pattern that becomes noticable when the
% arrangements are written out in increasing order.  The base cases for one ball and for
% one bin are trivial, and the rest of the cases can be formed by induction in two directions.

for row = 2:ballCount,
    for column = 2:binCount,

        % Increment the first column of the case for one less ball.
        temp1 = outputCellArray2D{row-1,column};
        temp1(:,1) = temp1(:,1) + 1;        
        
        % Prepend a column of zeros to the case for one less bin.
        temp2 = outputCellArray2D{row, column-1};
        temp2 = [zeros(size(temp2,1),1) temp2];  

        % The case for one more bin is just the vertical concatenation.
        outputCellArray2D{row,column} = [temp1; temp2];
    end
end

% Pack the outputs
if nargin == 2,
    outputCellArray = outputCellArray2D{end,end};
else
    outputCellArray = outputCellArray2D(:,end);
end
