% Found some additional documentation that showed people only keeping
% most frequent or "most important" terms for each document, and then
% S. Lafon 2006 uses a "conditional entropy" cutoff instead of the
% "standard" entropy that I'd assumed Mauro was talking about.

clear all;

load('/Users/emonson/X_Archives/EMoDocAnalysis/X_Data/SelfTokenizedTDMs/tdm_emo1nvj_112509.mat');

labels_self = dlmread('/Users/emonson/X_Archives/EMoDocAnalysis/X_Data/SelfTokenizedTDMs/pure.classes');
classes_self = labels_self(:,[2 1]);

% Get rid of any "non-pure" classes (assigned 0 in X20)
%   by using filename integer converted to array index
%   Note: this takes care of missing 10976 file...
tdm_self = tdm(:,classes_self(:,2)-10000+1);
terms_self = terms';

clear('tdm', 'labels_self', 'terms', 'names');

%% Original "magic" SciNews data set

load('/Users/emonson/Data/Fodava/MauroDocAnalysis/Data/X20.mat');

I = X(classes(:,1)>0,:)';
for ii=1:size(I,2), 
    I(:,ii) = I(:,ii)./min(I(I(:,ii)>0,ii)); 
end;
tdm_orig = sparse(round(I));

classes_orig = classes(classes(:,1)>0,:);
terms_orig = dict;

clear('X', 'I', 'classes', 'dict');

% NOTE: at 879 the orig set of frequencies seem to be one doc ahead of
%   the self set...

%% Lookup Maps

term_idx_map_self = containers.Map(terms_self, 1:length(terms_self));
term_idx_map_orig = containers.Map(terms_orig, 1:length(terms_orig));

%% Plot some term frequencies across documents

freq_sum_orig = sum(tdm_orig,2);
[~,II] = sort(freq_sum_orig, 'descend');
trial_terms = terms_orig(II(1:10));
n_docs = size(tdm_orig,2);

figure; 
hold on;
for ii = 1:length(trial_terms),
    term = trial_terms{ii};
    if isKey(term_idx_map_self, term)
        plot(tdm_self(term_idx_map_self(trial_terms{ii}),:)+(ii-1)*10, 'r');
        text(n_docs+10, (ii-1)*10+3, '+');
    end
    if isKey(term_idx_map_orig, term)
        plot(tdm_orig(term_idx_map_orig(trial_terms{ii}),:)+(ii-1)*10, 'k');
        text(n_docs+20, (ii-1)*10+3, 'o');
    end
    text(n_docs+30, (ii-1)*10+3, trial_terms{ii});
end

%% TFIDF (still sparse)

tfidf_self = tdm_tfidf(tdm_self);
tfidf_orig = tdm_tfidf(tdm_orig);

%% Entropy

h_cond_self = tdm_conditional_entropy(tfidf_self);
h_cond_orig = tdm_conditional_entropy(tfidf_orig);

h_norm_self = tdm_normal_entropy(tfidf_self);
h_norm_orig = tdm_normal_entropy(tfidf_orig);

figure;
subplot(1,2,1);
hist(h_cond_self, 100);
title('self conditional entropy');
subplot(1,2,2);
hist(h_cond_orig, 100);
title('orig conditional entropy');

figure;
subplot(1,2,1);
hist(h_norm_self, 100);
title('self normal entropy');
subplot(1,2,2);
hist(h_norm_orig, 100);
title('orig normal entropy');


