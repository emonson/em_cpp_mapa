function tdm_norm = tdm_tfidf(tdm)

% calculate TFIDF (std) normalization for word counts
nkj = sum(tdm,1)';      % how many terms in each document
D = size(tdm,2);        % number of documents
df = sum(tdm>0,2);      % number of documents each term shows up in
idf = log(D./(1+df));   % the 1+ is common to avoid divide-by-zero

[ii,jj,vv] = find(tdm);
vv_norm = (vv./nkj(jj)).*idf(ii);

tdm_norm = sparse(ii,jj,vv_norm);

end