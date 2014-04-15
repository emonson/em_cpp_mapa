function averageBasisError = average_basis_error(bases1, bases2);

basisNumber = length(bases1);
if length(bases2)~=basisNumber
    error('The number of bases must match.');
end

averageBasisError = 0;
for basisIndex=1:basisNumber
    averageBasisError = averageBasisError + subspace_angle(bases1{basisIndex}, bases2{basisIndex});
end

averageBasisError = averageBasisError/basisNumber/pi*180;