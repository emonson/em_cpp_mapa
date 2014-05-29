
load('/Users/emonson/Data/Fodava/EMoDocMatlabData/n20_sub4_tdm_train.mat');

% all "non-zero class" documents (need D x N)

% calculate TFIDF (std) normalization for word counts
nkj = sum(tdm,1)';      % how many terms in each document
D = size(tdm,2);        % number of documents
df = sum(tdm>0,2);      % number of documents each term shows up in
idf = log(D./(1+df));   % the 1+ is common to avoid divide-by-zero

[ii,jj,vv] = find(tdm);
vv_norm = (vv./nkj(jj)).*idf(ii);

tdm_norm = sparse(ii,jj,vv_norm);
I = full(tdm_norm);

% Take cosine similarity
% I is D x N
I2 = sqrt(sum(I.*I, 1));
good_cols = I2 > 0;
I = I(:, good_cols);
I2 = I2(good_cols);
labels_true = double(labels(good_cols)');
cos_sim = (I'*I)./(I2'*I2);

[U, S, V] = svd(I);
% [U, S, V] = svd(cos_sim,0);
red_dim = 50;

X = V(:,1:red_dim)*S(1:red_dim,1:red_dim); % 1047 points in 8 true classes
opts = struct('dmax', 6, 'Kmax', 6, 'n0', length(labels_true), 'plotFigs', false);
% opts = struct('K', 2, 'n0', 1177, 'plotFigs', true);
% X = I';
% opts = struct('dmax', 12, 'Kmax', 64, 'n0', 1047, 'plotFigs', true);
figure; do_plot_data(X(:,1:3));

fprintf('Running MAPA\n');
tic; 
[m_labels, planeDims, planeCenters, planeBases] = mapa_min(X,opts); 
fprintf(1,'Time Used: %3.2f\n', toc);

% Plot category assignments with some jitter
figure; plot(labels_true+0.15*randn(length(labels_true),1),m_labels+0.15*randn(size(m_labels)),'ko','Color',[0.4 0 0]);
xlabel('True categories');
ylabel('Assigned plane index');
[MisclassificationRate, counts_mtx, opt_perm] = clustering_error_improved(m_labels,labels_true);
disp(['Misclassification rate: ' num2str(MisclassificationRate)]);
figure; 
imagesc(counts_mtx); 
axis image;
colormap(gray);
caxis([0 max(counts_mtx(:))]);

%% Major terms
% figure;
% hold on;
for ii=1:length(planeCenters), 
    % Need to reproject vectors back into term space
    cent = abs(planeCenters{ii}*U(:,1:red_dim)');
    [YY,II]=sort(cent, 'descend'); 
    disp(ii); 
    disp(terms(II(1:20))); 
    % plot(cent(II));
end

