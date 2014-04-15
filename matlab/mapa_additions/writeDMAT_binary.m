function [n_elements] = writeDMAT_binary(filename, matrix)
% Write out a matrix in IGL DMAT binary

% doing combination of ascii then binary, so just 'w'
fid = fopen(filename, 'w');

% binary file format needs zeros in the first line
fprintf(fid, '0 0\n');
% for some reason igl::readDMAT() and write do n_cols n_rows order
fprintf(fid,'%d %d\n', size(matrix,2), size(matrix,1));
% the rest write binary
n_elements = fwrite(fid, matrix, 'double');

fclose(fid);

end