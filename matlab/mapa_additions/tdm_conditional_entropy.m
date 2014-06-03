function h2 = tdm_conditional_entropy(tdm)

% S. Lafon 2006 uses a "conditional entropy" cutoff instead of the
% "standard" entropy that I'd assumed Mauro was talking about.
% Calculating "conditional entropy"
sum2 = sum(tdm,2);
ratio2 = tdm./repmat(sum2,[1 size(tdm,2)]);
lratio2 = log(ratio2);
lratio2(isinf(lratio2)) = 0;
h2 = -1*sum(ratio2.*lratio2,2);

end