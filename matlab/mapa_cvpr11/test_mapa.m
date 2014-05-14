
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To run this script, the following needs to be done first:
% 1. Download the 'GPCA-voting' package at 
%     http://perception.csl.uiuc.edu/software/GPCA/gpca-voting.tar.gz.
%    The function 'generate_samples.m' is used to creat synthetic data.
% 2. Download the real data used in the paper at 
%    http://www.math.duke.edu/~glchen/mapa_realdata.zip,
%    and save it in a subfolder with the name 'data'.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear all;
% close all;
% clc;

%% pick an experiment
pExampleNames  = {'Artificial_Data','Artificial_Data_Preset','Simulation_Mode','Motion_Segmentation','Face_Clustering',...
    'Medical12_features','Medical12images','CorelImages_features','CorelImages',...
    'SciNews_TDM', 'SciNews_TFcorr', 'SciNews_TF', 'n20_sub1', 'SciNews_TFIDF', 'SciNews_LDA',...
    'SciNews_Cat1'};

fprintf('\n Examples:\n');
for k = 1:length(pExampleNames),
    fprintf('\n [%d] %s',k,pExampleNames{k});
end;
fprintf('\n\n  ');

pExampleIdx = input('Pick an example to run: \n');

%%
switch pExampleNames{pExampleIdx}
    
    case 'Artificial_Data'
        
        % set model parameters
        isaffine = false; %affine subspaces or linear
        D = 3; % ambient dimension
        aprioriSubspaceDimensions = [1 2 1];
        K = numel(aprioriSubspaceDimensions); % number of subspaces
        d_max = max(aprioriSubspaceDimensions);
        noiseLevel = 0.04/sqrt(D);
        groupSizes = 200*ones(1,K); %100*aprioriSubspaceDimensions;
        N = sum(groupSizes); %data size
        
        % creat linear subspaces. The function generate_samples is borrowed
        % from the GPCA-voting package at the following url:
        % http://perception.csl.uiuc.edu/software/GPCA/gpca-voting.tar.gz
        [Xt, aprioriSampleLabels, aprioriGroupBases] = generate_samples(...
            'ambientSpaceDimension', D,...
            'groupSizes', groupSizes,...
            'basisDimensions', aprioriSubspaceDimensions,...
            'noiseLevel', noiseLevel,...
            'noiseStatistic', 'gaussian', ...
            'isAffine', 0,...
            'outlierPercentage', 0, ...
            'minimumSubspaceAngle', pi/6);
        X = Xt'; % Xt is D-by-N, X is N-by-D
        
        % creat affine subspaces
        if isaffine
            randomCenters = random('norm', 0, 1, K, D);
            matCenters = zeros(N,D);
            for k = 1:length(aprioriSubspaceDimensions)
                matCenters(1+sum(groupSizes(1:k-1)): sum(groupSizes(1:k)),:) = repmat(randomCenters(k,:), groupSizes(k), 1);
            end
            X = X + matCenters;
        end
        
        % set mapa parameters
        opts = struct('n0',20*K, 'dmax',d_max, 'Kmax',2*K, 'plotFigs',true, 'showSpectrum', 0);
                
        % apply mapa
        tic; [labels, planeDims, other] = mapa(X,opts); TimeUsed = toc
        MisclassificationRate = clustering_error(labels,aprioriSampleLabels)
        
   case 'Artificial_Data_Preset'
       
       % preset data from a sample run of Artificial_Data for testing
       % (avoid randomness)
       load('artificial_data_rev1.mat');
       
       % make sure using the same seeds rather than randomly generated
       opts.seeds = other.seeds;
       % opts.discardCols = 0.2;
       % opts.discardRows = 0.2;
       
       % DEBUG setting K
       opts = rmfield(opts, 'Kmax');
       opts.K = 3;
       
       % apply mapa
       [labels, planeDims] = mapa_min(X,opts);
       MisclassificationRate = clustering_error(labels,aprioriSampleLabels);
       disp(MisclassificationRate);
        
   case 'Simulation_Mode'
        
        nLoops = 10;

        D = 3
        dims = [1 2 1]
        noiseLevel = 0.04/sqrt(D);
        
        run_simulation_mapa_only(D, dims, noiseLevel, nLoops);
        
    case 'Motion_Segmentation'
        
        dataset = {'kanatani1', 'kanatani2','kanatani3'}; 
        
        for i = 1:3
            
            sequence = i
            eval(['load /Users/emonson/Data/MAPA/MotionSegmentation/' dataset{i} '/' dataset{i} '_truth']);
            [~, N, F] = size(x);
            z = transpose(reshape(permute(x(1:2,:,:),[1 3 2]),2*F,N)); 
            [U,S] = svds(z-repmat(mean(z,1),N,1),10);
            X = U(:,1:10)*S(1:10,1:10);
            [aprioriSampleLabels, inds] = sort(s); 
            X = X(inds,:); %figure; do_plot_data(X(:,1:3))
            
            opts = struct('dmax',3, 'Kmax',5, 'n0',N, 'plotFigs',true);
            tic; [labels, planeDims, other] = mapa(X,opts); TimeUsed = toc
            planeDims
            estimatedTolerance = other.eps*sqrt(10)
            MisclassificationRate = clustering_error(labels,aprioriSampleLabels)
            
            fprintf('\n')
            
        end
        
    case 'Face_Clustering'
        
        load /Users/emonson/Data/Fodava/MAPA/FaceClustering/yaleFacesB
        
%         % faces 5, 8, 10
%         [U, S, V] = svd(I(:, [64*4+1:320 64*7+1:64*8 64*9+1:640]),0);
%         X = V(:,1:10)*S(1:10,1:10); N = size(X,1);
%         opts = struct('dmax',3, 'Kmax',6, 'n0',N, 'plotFigs',true, 'MinNetPts',6, 'nScales',30, 'nPtsPerScale',3);
%         tic; [labels, planeDims, other] = mapa(X,opts); TimeUsed = toc
%         MisclassificationRate = clustering_error(labels,reshape(repmat([1 2 3], 64, 1), 1, []))
        
        % all 10 faces
        [U, S, V] = svd(I,0);
        X = V(:,1:30)*S(1:30,1:30); % 640 points
        figure; do_plot_data(X(:,1:3));
        opts = struct('dmax',3, 'Kmax',15, 'n0',640, 'plotFigs',true);
        tic; [labels, planeDims, other] = mapa(X,opts); TimeUsed = toc;
        MisclassificationRate = clustering_error(labels,reshape(repmat(1:10, 64, 1), 1, []));
        
    case 'Medical12_features'
        
        cd('/Users/emonson/Data/Ronak');
        
        I = csvread('Medical12Classes_data.csv');
        classes = csvread('Medical12Classes_classes.csv');
        
        opts = struct('dmax', 3, 'Kmax', 20, 'n0', 540, 'plotFigs', true);
        tic; 
        [labels, planeDims, other] = mapa(I, opts); 
        TimeUsed = toc;
        figure; 
        plot(classes+0.15*randn(size(classes)),labels+0.15*randn(size(labels)),'ko','Color',[0.4 0 0]);
        xlabel('True categories');
        ylabel('Assigned plane index');
       
    case 'Medical12images'

        cd('/Users/emonson/Data/Ronak');

        % I = csvread('Medical12Classes_data.csv');
        classes = csvread('Medical12Classes_classes.csv');
        fid = fopen('Medical12Classes_filenames.csv');
        C = textscan(fid,'%s');
        filenames = C{1};
        fclose(fid);

        % randomize order
        idxs = randperm(length(filenames));
        filenames = filenames(idxs);
        classes = classes(idxs);

        cd('Medical12Classes');
        tmp = imread(filenames{1});
        [imR,imC] = size(tmp);

        X0 = zeros(imR*imC, length(filenames));
        for ii = 1:length(filenames),
            tmp = imread(filenames{ii});
            X0(:,ii) = double(reshape(tmp, imR*imC, 1));
        end;

        [U, S, V] = svd(X0,0);
        X = V(:,1:30)*S(1:30,1:30); 
        opts = struct('dmax', 3, 'Kmax', 30, 'n0', 540, 'plotFigs', true);
        tic; 
        [labels, planeDims, other] = mapa(X, opts); 
        TimeUsed = toc;
        figure; 
        plot(classes+0.15*randn(size(classes)),labels+0.15*randn(size(labels)),'ko','Color',[0.4 0 0]);
        xlabel('True categories');
        ylabel('Assigned plane index');


    case 'CorelImages_features'
        
        cd('/Users/emonson/Data/Ronak');
        
        I = csvread('ImagesCorel_data.csv')';
        classes = csvread('ImagesCorel_classes.csv');
        
        % [U, S, V] = svd(I,0);
        % X = V(:,1:50)*S(1:50,1:50); % 640 points
        opts = struct('dmax', 3, 'Kmax', 50, 'n0', 1000, 'plotFigs', true);
        tic; 
        [labels, planeDims, other] = mapa(I', opts); 
        TimeUsed = toc;
        figure; 
        plot(classes+0.15*randn(size(classes)),labels+0.15*randn(size(labels)),'ko','Color',[0.4 0 0]);
        xlabel('True categories');
        ylabel('Assigned plane index');
    
    case 'CorelImages'
        cd('/Users/emonson/Data/Ronak');

        % I = csvread('ImagesCorel_data.csv');
        classes = csvread('ImagesCorel_classes.csv');
        fid = fopen('ImagesCorel_filenames.csv');
        C = textscan(fid,'%s');
        filenames = C{1};
        fclose(fid);

        % randomize order
        idxs = randperm(length(filenames));
        filenames = filenames(idxs);
        classes = classes(idxs);

        cd('ImagesCorel');
        tmp = imread(filenames{1});
        [imR,imC,c] = size(tmp);
        if (imR < imC),
            tmp = permute(tmp,[2 1 3]);
        end;
        % Trying to compensate for 3-color images
        [imR,imC,c] = size(tmp);
        imC = 3*imC;

        dataset = struct();
        dataset.N = length(filenames);
        dataset.projectionDimension = 0;

        X0 = zeros(imR*imC, length(filenames));
        for ii = 1:length(filenames),
            tmp = imread(filenames{ii});
            [r,c,cc] = size(tmp);
            if (r < c),
                tmp = permute(tmp,[2 1 3]);
            end;
           X0(:,ii) = double(reshape(tmp, imR*imC, 1));
        end;

        [U, S, V] = svd(X0,0);
        X = V(:,1:150)*S(1:150,1:150); 
        opts = struct('dmax', 3, 'Kmax', 50, 'n0', 1000, 'plotFigs', true);
        tic; 
        [labels, planeDims, other] = mapa(X, opts); 
        TimeUsed = toc;
        figure; 
        plot(classes+0.15*randn(size(classes)),labels+0.15*randn(size(labels)),'ko','Color',[0.4 0 0]);
        xlabel('True categories');
        ylabel('Assigned plane index');

 
    case 'SciNews_TDM'
        
        load('/Users/emonson/Data/Fodava/MauroDocAnalysis/Data/X20.mat');
        
        % all "non-zero class" documents
        I = X(classes(:,1)>0,:)';
        labels_true = classes(classes(:,1)>0,1);
        clear('X', 'classes');
        
        fprintf('Computing SVD\n');
        % try moving back to counts instead of normalized freq/doc
        for ii=1:1047, I(:,ii) = I(:,ii)./min(I(I(:,ii)>0,ii)); end;
        I = round(I);
        
        [U, S, V] = svd(I,0);
        X = V(:,1:50)*S(1:50,1:50); % 1047 points in 8 true classes
        figure; do_plot_data(X(:,1:3));
        opts = struct('dmax', 6, 'Kmax', 32, 'n0', 1047, 'plotFigs', true);
        fprintf('Running MAPA\n');
        tic; [labels, planeDims, other] = mapa(X,opts); 
        fprintf(1,'Time Used: %3.2f\n', toc);
        % Plot category assignments with some jitter
        figure; plot(labels_true+0.15*randn(size(labels_true)),labels+0.15*randn(size(labels)),'ko','Color',[0.4 0 0]);
        xlabel('True categories');
        ylabel('Assigned plane index');
        % MisclassificationRate = clustering_error(labels,reshape(repmat(1:10, 64, 1), 1, []))

    case 'SciNews_TFcorr'
        
        load('/Users/emonson/Data/Fodava/MauroDocAnalysis/Data/X20.mat');
        
        % all "non-zero class" documents
        I = X(classes(:,1)>0,:)';
        labels_true = classes(classes(:,1)>0,1);
        clear('X', 'classes');
        
        fprintf('Computing TF corr matrix\n');
        % try moving back to counts instead of normalized freq/doc
        for ii=1:1047, I(:,ii) = I(:,ii)./min(I(I(:,ii)>0,ii)); end; 
        tdm = round(I);
 
        % calculate TF (std) normalization for word counts
        nkj = sum(tdm,1)';      % how many terms in each document

        [ii,jj,vv] = find(tdm);
        vv_norm = (vv./nkj(jj));

        tdm_norm = sparse(ii,jj,vv_norm);
 
        corrQuantile = 0.90;    % Global cutoff for correlations
        NN = 5;                 % number of nn required in corr mtx 
                                % -- includes one self-neighbor
                                
        XX = mat_corr(tdm_norm);

        fprintf(1,'\tCalculating %f quantile of correlation values',corrQuantile); toc;
        qq = quantile(XX(XX>0),corrQuantile);

        fprintf(1,'\tFiltering out low corr values :: '); toc;
        YY = sparse(XX.*(XX>qq));

        % Check if any rows/columns of YY are too sparse
        fprintf(1,'\tAdjusting neighbors :: '); toc;
        for ii = find(sum(YY>0,1) < NN),
            % Add elements from XX back into YY to reach required NN count
            [sC,sI] = sort(XX(:,ii),'descend');
            YY(sI(1:NN),ii) = sC(1:NN);
            YY(ii,sI(1:NN)) = sC(1:NN);
        end;

        numConnComp = graphconncomp(YY);
        fprintf(1,'\tNumber of connected components = %d\n', numConnComp);

        % For now, break if graph not completely connected...
        if (numConnComp > 1)
            break;
        end;

        clear('XX');
        
        I = full(YY);
        
        % Trying sorted to see structure of matA better...
        [~,II] = sort(labels_true);
        I = I(:,II);
        labels_true = labels_true(II);
        
        fprintf('Computing SVD\n');
        [U, S, V] = svd(I,0);
        X = V(:,1:50)*S(1:50,1:50); % 1047 points in 8 true classes
        opts = struct('dmax', 6, 'Kmax', 32, 'n0', 1047, 'plotFigs', true);
        % X = I';
        % opts = struct('dmax', 12, 'Kmax', 64, 'n0', 1047, 'plotFigs', true);
        figure; do_plot_data(X(:,1:3));
        fprintf('Running MAPA\n');
        tic; [labels, planeDims, other] = mapa(X,opts); 
        fprintf(1,'Time Used: %3.2f\n', toc);
        % Plot category assignments with some jitter
        figure; plot(labels_true+0.15*randn(size(labels_true)),labels+0.15*randn(size(labels)),'ko','Color',[0.4 0 0]);
        xlabel('True categories');
        ylabel('Assigned plane index');
        % MisclassificationRate = clustering_error(labels,reshape(repmat(1:10, 64, 1), 1, []))

    case 'SciNews_TF'
        
        load('/Users/emonson/Data/Fodava/MauroDocAnalysis/Data/X20.mat');
        
        % all "non-zero class" documents
        I = X(classes(:,1)>0,:)';
        labels_true = classes(classes(:,1)>0,1);
        clear('X', 'classes');
        
        fprintf('Computing SVD\n');
        % try moving back to counts instead of normalized freq/doc
        for ii=1:1047, I(:,ii) = I(:,ii)./min(I(I(:,ii)>0,ii)); end; 
        tdm = round(I);
 
        % calculate TF (std) normalization for word counts
        nkj = sum(tdm,1)';      % how many terms in each document

        [ii,jj,vv] = find(tdm);
        vv_norm = (vv./nkj(jj));

        tdm_norm = sparse(ii,jj,vv_norm);
        I = full(tdm_norm);
        
        % Trying sorted to see structure of matA better...
        % [YY,II] = sort(labels_true);
        % I = I(:,II);
        % labels_true = labels_true(II);
        
        nPlanes = 8;
        
        [U, S, V] = svd(I,0);
        X = V(:,1:50)*S(1:50,1:50); % 1047 points in 8 true classes
        % opts = struct('dmax', 6, 'Kmax', 32, 'n0', 1047, 'plotFigs', true);
        % opts = struct('dmax', 12, 'Kmax', nPlanes, 'n0', 1047, 'plotFigs', false);
        opts = struct('K', nPlanes, 'n0', 1047, 'plotFigs', false);
        % X = I';
        % opts = struct('dmax', 12, 'Kmax', 64, 'n0', 1047, 'plotFigs', true);
        % figure; do_plot_data(X(:,1:3));
        fprintf('Running MAPA\n');
        tic; [labels, planeDims, other] = mapa(X,opts); 
        fprintf(1,'Time Used: %3.2f\n', toc);
        % Plot category assignments with some jitter
        figure; plot(labels_true+0.15*randn(size(labels_true)),labels+0.15*randn(size(labels)),'ko','Color',[0.4 0 0]);
        xlabel('True categories');
        ylabel('Assigned plane index');
        % MisclassificationRate = clustering_error(labels,reshape(repmat(1:10, 64, 1), 1, []))

        % Take single class subset
        centers = cell(1,nPlanes);
        bases = cell(1,nPlanes);
        [cc,bb] = computing_bases(X, labels, planeDims);
        for ii = 1:nPlanes,
            centers{ii} = cc{ii}';
            bases{ii} = bb{ii}';
        end
        
        ang = zeros(nPlanes);
        for ii = 1:nPlanes, 
            for jj = 1:nPlanes, 
                if(ii~=jj), 
                    ang(ii,jj) = max(acos(svd(bases{ii}'*bases{jj})))*360/(2*pi); 
                end; 
            end; 
        end;
        figure;
        imagesc(ang);
        axis image;
        colormap(hot);
        caxis([60 90]);
        colorbar();
        
        distances_in_planes = cell(nPlanes,nPlanes);
        distances_to_planes = cell(nPlanes,nPlanes);
        for ii = 1:nPlanes,
            Xsub = X(labels == ii,:)';
            nn = size(Xsub,2);
            in_plane_dist = zeros(nn,1);
            to_plane_dist = zeros(nn,1);
            for jj = 1:nPlanes,
                for kk = 1:nn, 
                    in_plane_dist(kk) = norm(bases{jj}*bases{jj}'*(Xsub(:,kk)-centers{jj})); 
                    to_plane_dist(kk) = norm(Xsub(:,kk) - bases{jj}*bases{jj}'*(Xsub(:,kk)-centers{jj}) - centers{jj}); 
                end;
                distances_in_planes{ii,jj} = in_plane_dist;
                distances_to_planes{ii,jj} = to_plane_dist;
            end
        end;
        
        distances_cats_in_planes = cell(8,nPlanes);
        distances_cats_to_planes = cell(nPlanes,nPlanes);
        for ii = 1:8,
            Xsub = X(labels_true == ii,:)';
            nn = size(Xsub,2);
            in_plane_dist = zeros(nn,1);
            to_plane_dist = zeros(nn,1);
            for jj = 1:nPlanes,
                for kk = 1:nn, 
                    in_plane_dist(kk) = norm(bases{jj}*bases{jj}'*(Xsub(:,kk)-centers{jj})); 
                    to_plane_dist(kk) = norm(Xsub(:,kk) - bases{jj}*bases{jj}'*(Xsub(:,kk)-centers{jj}) - centers{jj}); 
               end;
                distances_cats_in_planes{ii,jj} = in_plane_dist;
                distances_cats_to_planes{ii,jj} = to_plane_dist;
           end
        end;
        
        % Plot with distance distributions to center (k)
        % and distances from this to other centers (r) for each category
%         colors = brewerDark1(nPlanes);
%         figure;
%         hold on;
%         for ii = 1:nPlanes,
%            % plot(ii + 0.025*randn(size(centers_dist(:,ii))), centers_dist(:,ii), 'k.');
%            for jj = 1:nPlanes,
%              plot(ii + 0.05*randn(size(distances_planes{ii,jj})), distances_planes{ii,jj}, 'ko', 'Color', colors(jj,:));
%            end
%         end;
%         set(gca, 'XGrid', 'on');
%         figure; imagesc(1:nPlanes); colormap(brewerDark1);
        
        % Plot small multiples box plots of distances distributions
        % MAPA plane assignments
        figure;
        for ii = 1:nPlanes,
            subplot(ceil(nPlanes/2),2,ii);
            dmat = cell2mat(distances_in_planes(ii,:));
            plot(dmat','Color',[0.8 0.8 0.8]);
            hold on;
            boxplot(dmat);
            title(['plane ' int2str(ii)]);
            xlabel('plane');
            ylabel('distance');
            ylim([0 0.12]);
        end
        
        % Plot small multiples box plots of distances to planes distributions
        % MAPA plane assignments
        figure;
        for ii = 1:nPlanes,
            subplot(ceil(nPlanes/2),2,ii);
            dtmat = cell2mat(distances_to_planes(ii,:));
            plot(dtmat','Color',[0.8 0.8 0.8]);
            hold on;
            boxplot(dtmat);
            title(['plane ' int2str(ii)]);
            xlabel('plane');
            ylabel('distance');
            % ylim([0 0.12]);
        end
        
        % Plot small multiples box plots of distance distributions
        % Real category assignments
        figure;
        for ii = 1:8,
            subplot(4,2,ii);
            plot(cell2mat(distances_cats_in_planes(ii,:))','Color',[0.8 0.8 0.8]);
            hold on;
            dmat = cell2mat(distances_cats_in_planes(ii,:));
            boxplot(dmat);
            title(['category ' int2str(ii)]);
            xlabel('plane');
            ylabel('distance');
            ylim([0 0.12]);
        end
        
        % Plot small multiples box plots of distance to plane distributions
        % Real category assignments
        figure;
        for ii = 1:8,
            subplot(4,2,ii);
            plot(cell2mat(distances_cats_to_planes(ii,:))','Color',[0.8 0.8 0.8]);
            hold on;
            dmat = cell2mat(distances_cats_to_planes(ii,:));
            boxplot(dmat);
            title(['category ' int2str(ii)]);
            xlabel('plane');
            ylabel('distance');
            % ylim([0 0.12]);
        end
        

    
    case 'n20_sub1'
        
        load('/Users/emonson/Data/Fodava/EMoDocMatlabData/n20_sub1train_TFcorr_111809.mat');
        
        % all "non-zero class" documents (need D x N)
        I = full(G.W);
        labels_true = double(labels');
        
        % Trying sorted to see structure of matA better...
        % [YY,II] = sort(labels);
        % I = I(:,II);
        % labels_true = double(labels(II))';
        
        [U, S, V] = svd(I,0);
        X = V(:,1:50)*S(1:50,1:50); % 1047 points in 8 true classes
        opts = struct('dmax', 6, 'Kmax', 32, 'n0', 1177, 'plotFigs', true);
        % opts = struct('K', 2, 'n0', 1177, 'plotFigs', true);
        % X = I';
        % opts = struct('dmax', 12, 'Kmax', 64, 'n0', 1047, 'plotFigs', true);
        figure; do_plot_data(X(:,1:3));
        fprintf('Running MAPA\n');
        tic; [m_labels, planeDims, other] = mapa(X,opts); 
        fprintf(1,'Time Used: %3.2f\n', toc);
        % Plot category assignments with some jitter
        figure; plot(labels_true+0.15*randn(size(labels_true)),m_labels+0.15*randn(size(m_labels)),'ko','Color',[0.4 0 0]);
        xlabel('True categories');
        ylabel('Assigned plane index');
        % MisclassificationRate = clustering_error(labels,reshape(repmat(1:10, 64, 1), 1, []))

    case 'SciNews_TFIDF'
        
        load('/Users/emonson/Data/Fodava/MauroDocAnalysis/Data/X20.mat');
        
        % all "non-zero class" documents
        I = X(classes(:,1)>0,:)';
        labels_true = classes(classes(:,1)>0,1);
        clear('X', 'classes');
        
        fprintf('Computing SVD\n');
        % try moving back to counts instead of normalized freq/doc
        for ii=1:1047, I(:,ii) = I(:,ii)./min(I(I(:,ii)>0,ii)); end;
        tdm = round(I);
 
        % calculate TFIDF (std) normalization for word counts
        nkj = sum(tdm,1)';      % how many terms in each document
        D = size(tdm,2);        % number of documents
        df = sum(tdm>0,2);      % number of documents each term shows up in
        idf = log(D./(1+df));   % the 1+ is common to avoid divide-by-zero

        [ii,jj,vv] = find(tdm);
        vv_norm = (vv./nkj(jj)).*idf(ii);

        tdm_norm = sparse(ii,jj,vv_norm);
        I = full(tdm_norm);
        
        [U, S, V] = svd(I,0);
        X = V(:,1:50)*S(1:50,1:50); % 1047 points in 8 true classes
        figure; do_plot_data(X(:,1:3));
        opts = struct('dmax', 6, 'Kmax', 32, 'n0', 1047, 'plotFigs', true);
        fprintf('Running MAPA\n');
        tic; [labels, planeDims, other] = mapa(X,opts); 
        fprintf(1,'Time Used: %3.2f\n', toc);
        % Plot category assignments with some jitter
        figure; plot(labels_true+0.15*randn(size(labels_true)),labels+0.15*randn(size(labels)),'ko','Color',[0.4 0 0]);
        xlabel('True categories');
        ylabel('Assigned plane index');
        % MisclassificationRate = clustering_error(labels,reshape(repmat(1:10, 64, 1), 1, []))

    case 'SciNews_LDA'
        
        load('/Users/emonson/Data/Fodava/MAPA/SciNews_LDA/sn_50.mat');
        load('/Users/emonson/Data/Fodava/MauroDocAnalysis/Data/X20.mat');
       
        % There is one extra document in X20 (idx = 977)
        [~,II] = setdiff(classes(:,2), doc_names);
        classes(II,:) = [];
        
        % all "non-zero class" documents
        I = doc_topics(classes(:,1)>0,:)';
        labels_true = classes(classes(:,1)>0,1);
        clear('X', 'classes');
        
        % Trying sorted to see structure of matA better...
        % [~,II] = sort(labels_true);
        % I = I(:,II);
        % labels_true = labels_true(II);
        
        fprintf('Computing SVD\n');
        % [U, S, V] = svd(I,0);
        % X = V(:,1:50)*S(1:50,1:50); % 1047 points in 8 true classes
        X = I';
        figure; 
        do_plot_data(X(:,1:3));
        drawnow;

        opts = struct('dmax', 6, 'Kmax', 32, 'n0', 1047, 'plotFigs', true);
        fprintf('Running MAPA\n');
        tic; [labels, planeDims, other] = mapa(X,opts); 
        fprintf(1,'Time Used: %3.2f\n', toc);
        % Plot category assignments with some jitter
        figure; plot(labels_true+0.15*randn(size(labels_true)),labels+0.15*randn(size(labels)),'ko','Color',[0.4 0 0]);
        xlabel('True categories');
        ylabel('Assigned plane index');
        % MisclassificationRate = clustering_error(labels,reshape(repmat(1:10, 64, 1), 1, []))

	case 'SciNews_Cat1'
        
        load('/Users/emonson/Data/Fodava/MauroDocAnalysis/Data/X20.mat');
        
        % all "non-zero class" documents
        I = X(classes(:,1)>0,:)';
        labels_true = classes(classes(:,1)>0,1);
        clear('X', 'classes');

        [U, S, V] = svd(I,0);
        X = V(:,1:30)*S(1:30,1:30); % 1047 points in 8 true classes
        
        % Take single class subset
        centers = cell(1,8);
        bases = cell(1,8);
        for subN = unique(labels_true)',
            
            Xsub = X(labels_true == subN, :);
            labels_sub = labels_true(labels_true == subN);

            opts = struct('K', 1, 'n0', size(Xsub, 2), 'plotFigs', false);
            % opts = struct('dmax', 12, 'Kmax', 64, 'n0', 1047, 'plotFigs', true);

            fprintf('Running MAPA, cat = %d\n', subN);

            [labels, planeDims, other] = mapa(Xsub, opts); 
            [cc,bb] = computing_bases(Xsub, labels, planeDims);
            centers{subN} = cc{1}';
            bases{subN} = bb{1}';
            
        end
        
        ang = zeros(8);
        for ii = 1:8, 
            for jj = 1:8, 
                if(ii~=jj), 
                    ang(ii,jj) = max(acos(svd(bases{ii}'*bases{jj})))*360/(2*pi); 
                end; 
            end; 
        end;
        
        centers_dist = zeros(8);
        for ii=1:8, 
            for jj=1:8, 
                centers_dist(ii,jj) = norm(centers{ii} - centers{jj}); 
            end; 
        end;

        distances = cell(8,8);
        for ii = 1:8,
            Xsub = X(labels_true == ii,:)';
            % Xsub = X(labels == ii,:)';
            nn = size(Xsub,2);
            dd = zeros(nn,1);
            for jj = 1:8,
                for kk = 1:nn, 
                    dd(kk) = norm((bases{jj}*bases{jj}')*(Xsub(:,kk)-centers{jj})); 
                end;
                distances{ii,jj} = dd;
            end
        end;
        
        % Plot with distance distributions to center (k)
        % and distances from this to other centers (r) for each category
        colors = brewerDark1(8);
        figure;
        hold on;
        for ii = 1:8,
           % plot(ii + 0.025*randn(size(centers_dist(:,ii))), centers_dist(:,ii), 'k.');
           for jj = 1:8,
             plot(ii + 0.05*randn(size(distances{ii,jj})), distances{ii,jj}, 'ko', 'Color', colors(jj,:));
           end
        end;
        set(gca, 'XGrid', 'on');
        figure; imagesc(1:8); colormap(brewerDark1);
        
%         cm = brewerDark1(8);
%         cm_l = brighten(cm, 0.6);
%         figure;
%         hold on;
%         for ii = 1:8,
%             nn = sum(labels_true==ii);
%             plot(repmat(1:30,nn,1)'+0.15*randn(nn,30)', X(labels_true==ii,:)', '.', 'Color', cm_l(ii,:)); 
%         end
%         for ii = 1:8,
%             nn = sum(labels_true==ii);
%             plot(centers{ii}, 'o', 'Color', cm(ii,:), 'LineWidth', 4, 'MarkerSize', 8); 
%         end
                
        % MisclassificationRate = clustering_error(labels,reshape(repmat(1:10, 64, 1), 1, []))


end
