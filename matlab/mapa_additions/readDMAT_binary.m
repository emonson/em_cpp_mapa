function [matrix] = readDMAT_binary(filename)
% Write out a matrix in IGL DMAT binary

% doing combination of ascii then binary, so just 'w'
fid = fopen(filename, 'r');

% binary file format needs zeros in the first line
shape = fscanf(fid, '%d %d\n', 2);
if ~all(shape==0),
    fprintf('Not a binary file\n');
    return;
end

% for some reason igl::readDMAT() and write do n_cols n_rows order
shape = fscanf(fid,'%d %d\n', 2);
n_cols = shape(1);
n_rows = shape(2);

% the rest write binary
matrix = fread(fid, n_cols * n_rows, 'double');
matrix = reshape(matrix, n_rows, n_cols);

fclose(fid);

end