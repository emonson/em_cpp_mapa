function h2 = tdm_conditional_entropy(tdm)

% Calculating "conditional entropy"
sum2 = sum(tdm,2);
ratio2 = tdm./repmat(sum2,[1 size(tdm,2)]);
lratio2 = log(ratio2);
lratio2(isinf(lratio2)) = 0;
h2 = -1*sum(ratio2.*lratio2,2);

end