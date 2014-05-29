% Found some additional documentation that showed people only keeping
% most frequent or "most important" terms for each document, and then
% S. Lafon 2006 uses a "conditional entropy" cutoff instead of the
% "standard" entropy that I'd assumed Mauro was talking about.

data_dir = '/Users/emonson/X_Archives/EMoDocAnalysis/X_Data/SelfTokenizedTDMs';
data_name = 'tdm_emo1nvj_112509.mat';
cd(data_dir);
load(data_name);

labels = dlmread('pure.classes');
classes = labels(:,[2 1]);

% Get rid of any "non-pure" classes (assigned 0 in X20)
%   by using filename integer converted to array index
%   Note: this takes care of missing 10976 file...
tdm = tdm(:,classes(:,2)-10000+1);

% % Only keeping most frequent terms from each document
% [Y,I] = sort(tdm,1,'descend');
% 
% tdm2 = sparse(size(tdm,1),size(tdm,2));
% for ii = 1:size(I,2),
%     tdm2(I(1:50,ii),ii) = tdm(I(1:50,ii),ii);
% end
% 
% % Getting rid of the rest of the terms
% throw_out = sum(tdm2,2)==0;
% tdm2(throw_out,:) = [];
% terms2 = terms;
% terms2(throw_out) = [];
% 
% % Normalizing resulting tdm
% sum1 = sum(tdm2,1);
% for ii = 1:size(tdm2,2),
%    tdm2(:,ii) = tdm2(:,ii)/sum1(ii); 
% end

%% Alternative starting point when not only keeping frequent terms
tdm2 = tdm;
terms2 = terms;

% Calculating "conditional entropy"
sum2 = sum(tdm,2);
numel2 = sum(tdm>0,2);
ratio2 = tdm./repmat(sum2,[1 size(tdm,2)]);
lratio2 = log(ratio2);
lratio2(isinf(lratio2)) = 0;
h2 = -1*sum(ratio2.*lratio2,2);

% Throwing out terms with high or low "conditional entropy"
throw_out = (h2<3.5|h2>5.0);
tdm2(throw_out,:) = [];
terms2(throw_out) = [];

% Normalizing resulting tdm
sum1 = sum(tdm2,1);
for ii = 1:size(tdm2,2),
   tdm2(:,ii) = tdm2(:,ii)/sum1(ii); 
end

% Calculating "normal" entropy
sum2b = sum(tdm2,2);
numel2b = sum(tdm2>0,2);
ratio2b = tdm2./repmat(sum2b,[1 size(tdm2,2)]);
lratio2b = log(ratio2b);
lratio2b(isinf(lratio2b)) = 0;
h2b = -1*sum(ratio2b.*lratio2b,2)./log(numel2b);
h2b(isnan(h2b)) = 0;

figure; hist(h2, 100);
figure; hist(h2b,100);
