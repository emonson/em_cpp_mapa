function h2b = tdm_normal_entropy(tdm2)

% Calculating "normal" entropy
sum2b = sum(tdm2,2);
numel2b = sum(tdm2>0,2);
ratio2b = tdm2./repmat(sum2b,[1 size(tdm2,2)]);
lratio2b = log(ratio2b);
lratio2b(isinf(lratio2b)) = 0;
h2b = -1*sum(ratio2b.*lratio2b,2)./log(numel2b);
h2b(isnan(h2b)) = 0;

end