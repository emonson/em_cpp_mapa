% Found some additional documentation that showed people only keeping
% most frequent or "most important" terms for each document, and then
% S. Lafon 2006 uses a "conditional entropy" cutoff instead of the
% "standard" entropy that I'd assumed Mauro was talking about.

% clear all;

load('/Users/emonson/Data/Fodava/SelfTokenizedTDMs/tdm_emo1nvj_112509.mat');

labels_self = dlmread('/Users/emonson/Data/Fodava/SelfTokenizedTDMs/pure.classes');
classes_self = labels_self(:,[2 1]);

% Get rid of any "non-pure" classes (assigned 0 in X20)
%   by using filename integer converted to array index
%   Note: this takes care of missing 10976 file...
tdm_self = tdm(:,classes_self(:,2)-10000+1);
terms_self = terms';

clear('tdm', 'labels_self', 'terms', 'names');

%% Original "magic" SciNews data set

load('/Users/emonson/Data/Fodava/MauroDocAnalysis/Data/X20.mat');

% I = X(classes(:,1)>0,:)';
I = X';
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

% freq_sum_orig = sum(tdm_orig,2);
% [~,II] = sort(freq_sum_orig, 'descend');
% trial_terms = terms_orig(II(1:10));
% n_docs = size(tdm_orig,2);
% 
% figure; 
% hold on;
% for ii = 1:length(trial_terms),
%     term = trial_terms{ii};
%     if isKey(term_idx_map_self, term)
%         plot(tdm_self(term_idx_map_self(trial_terms{ii}),:)+(ii-1)*10, 'r');
%         text(n_docs+10, (ii-1)*10+3, '+');
%     end
%     if isKey(term_idx_map_orig, term)
%         plot(tdm_orig(term_idx_map_orig(trial_terms{ii}),:)+(ii-1)*10, 'k');
%         text(n_docs+20, (ii-1)*10+3, 'o');
%     end
%     text(n_docs+30, (ii-1)*10+3, trial_terms{ii});
% end

%% TFIDF (still sparse)

tfidf_self = tdm_tfidf(tdm_self);
tfidf_orig = tdm_tfidf(tdm_orig);

%% Entropy

h_cond_self = tdm_conditional_entropy(tfidf_self);
h_cond_orig = tdm_conditional_entropy(tfidf_orig);

h_norm_self = tdm_normal_entropy(tfidf_self);
h_norm_orig = tdm_normal_entropy(tfidf_orig);

% figure;
% subplot(1,2,1);
% hist(h_cond_self, 100);
% xlim([0 8]);
% title('self conditional entropy');
% subplot(1,2,2);
% hist(h_cond_orig, 100);
% xlim([0 8]);
% title('orig conditional entropy');
% 
% figure;
% subplot(1,2,1);
% hist(h_norm_self, 100);
% xlim([0 1]);
% title('self normal entropy');
% subplot(1,2,2);
% hist(h_norm_orig, 100);
% xlim([0 1]);
% title('orig normal entropy');

%% Take conditional entropy and test MAPA clustering on quantile segments

filter_quantity = h_cond_self;
labels_true = classes_self(:,1);

ent_quantiles = quantile(filter_quantity, [0.0 0.33 0.67 1.0]);
red_dim = 50;

for ss = 1:length(ent_quantiles),
    if ss == 1,
        fprintf(1, '\nAll Segments\n');
        seg_rows = filter_quantity > 0;
    else
        fprintf(1, '\nSegment %d of %d\n', ss-1, length(ent_quantiles)-1);
        seg_rows = (filter_quantity >= ent_quantiles(ss-1)) & (filter_quantity < ent_quantiles(ss));
    end
    I = tdm_self(seg_rows, :);
    [U, S, V] = svds(I, red_dim);

    X = V * S; % 1047 points in 8 true classes
    opts = struct('dmax', 6, 'K', 8, 'n0', size(X,1), 'plotFigs', false);
    % opts = struct('K', 2, 'n0', 1177, 'plotFigs', true);
    % X = I';
    % opts = struct('dmax', 12, 'Kmax', 64, 'n0', 1047, 'plotFigs', true);
    figure; do_plot_data(X(:,1:3));

    fprintf('Running MAPA\n');
    tic; 
    [m_labels, planeDims, planeCenters, planeBases] = mapa_min(X,opts); 
    fprintf(1,'Time Used: %3.2f\n', toc);
    fprintf(1,'Plane Dims:\n');
    disp(planeDims);

    % Plot category assignments with some jitter
    K = length(planeDims);
    figure; plot(m_labels+0.15*randn(size(m_labels)), labels_true+0.15*randn(length(labels_true),1), 'ko','Color',[0.4 0 0]);
    title(['K = ' num2str(K) ', d_k = ' num2str(planeDims)]);
    xlabel('Assigned plane index');
    ylabel('True categories');
    [MisclassificationRate, counts_mtx, opt_perm] = clustering_error_improved(m_labels,labels_true);
    disp(['Misclassification rate: ' num2str(MisclassificationRate)]);
    figure; 
    imagesc(counts_mtx); 
    axis image;
    title(['K = ' num2str(K) ', d_k = ' num2str(planeDims)]);
    xlabel('Assigned plane index');
    ylabel('True categories');
    colormap(gray);
    caxis([0 max(counts_mtx(:))]);
end

