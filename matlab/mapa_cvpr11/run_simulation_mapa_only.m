function run_simulation_mapa_only(D, dims, noiseLevel, nExperiments)

warning off all

K = length(dims);
groupSizes = 200*ones(1,K); %100*dims;
N = sum(groupSizes);

%%
methods = {'REF', 'mapa', 'mapa_K'};
n_methods = length(methods);

p = zeros(2, nExperiments, n_methods);
t = zeros(2, nExperiments, n_methods);

planeDims_mapa = cell(2, nExperiments);
L2Errors_mapa = zeros(2, nExperiments);
eps = zeros(2, nExperiments);

%L2Errors_mapa_K = zeros(2, nExperiments);

%% mapa parameters
opts = struct( );
opts.dmax = max(dims);
opts.Kmax = K+1;
opts.n0 = 20*K;
%%
for i = 1:nExperiments
    
    if mod(i,5) == 0
        i
    end
    
    [Xt, aprioriSampleLabels] = generate_samples(...
        'ambientSpaceDimension', D,...
        'groupSizes', groupSizes,...
        'basisDimensions', dims,...
        'noiseLevel', noiseLevel,...
        'noiseStatistic', 'gaussian', ...
        'isAffine', 0,...
        'outlierPercentage', 0, ...
        'minimumSubspaceAngle', pi/6);
    
    X = Xt';
    
    %% linear
    
    % REF
    cnt = 1;
    tic;
    labels = K_flats(X, dims, aprioriSampleLabels, 1);
    t(1,i,cnt) = toc;
    p(1,i,cnt) = clustering_error(labels,aprioriSampleLabels);
    
    % mapa
    cnt = cnt+1;
    tic
    [labels, planeDims_mapa{1,i}, moreOutput] = mapa(X,opts);
    t(1,i,cnt) = toc;
    L2Errors_mapa(1,i) = moreOutput.L2Errors(numel(planeDims_mapa{1,i}));
    eps(1,i) = moreOutput.eps;
    p(1,i,cnt) = clustering_error(labels,aprioriSampleLabels);
    
    % mapa_K
    cnt = cnt+1;
    opts1 = rmfield(opts,'Kmax'); opts1.K = K;
    tic
    labels = mapa(X,opts1);
    t(1,i,cnt) = toc;
    %L2Errors_mapa_K(1,i) = moreOutput.L2Error;
    p(1,i,cnt) = clustering_error(labels,aprioriSampleLabels);

    %% affine
    randomCenters = random('norm', 0, 1/2, K, D);
    matCenters = zeros(N,D);
    for k = 1:K
        matCenters(1+sum(groupSizes(1:k-1)): sum(groupSizes(1:k)),:) = repmat(randomCenters(k,:), groupSizes(k), 1);
    end
    X = X + matCenters;

    % REF
    cnt = 1;
    tic;
    labels = K_flats(X, dims, aprioriSampleLabels,1);
    t(2,i,cnt) = toc;
    p(2,i,cnt) = clustering_error(labels,aprioriSampleLabels);
    
    % mapa
    cnt = cnt+1;
    tic
    [labels, planeDims_mapa{2,i}, moreOutput] = mapa(X,opts);
    t(2,i,cnt) = toc;
    L2Errors_mapa(2,i) = moreOutput.L2Errors(numel(planeDims_mapa{2,i}));
    eps(2,i) = moreOutput.eps;
    p(2,i,cnt) = clustering_error(labels,aprioriSampleLabels);
    
    % mapa_K
    cnt = cnt+1;
    tic
    labels = mapa(X,opts1);
    t(2,i,cnt) = toc;
    %L2Errors_mapa_K(2,i) = moreOutput.L2Error;
    p(2,i,cnt) = clustering_error(labels,aprioriSampleLabels);
     
end

p_L = modeling_success_rate(planeDims_mapa(1,:), dims);
p_A = modeling_success_rate(planeDims_mapa(2,:), dims);

%%
fprintf(1, 'percentage of correctly recovered models by mapa: \n')
[p_L; p_A]

fprintf(1, '\n percentage of correctly identified K by mapa:\n')
[sum(cellfun(@length, planeDims_mapa(1,:))==K)/nExperiments; sum(cellfun(@length, planeDims_mapa(2,:))==K)/nExperiments]

fprintf(1, '\n L2 error by mapa: ')
[mean(L2Errors_mapa(1,:)); mean(L2Errors_mapa(2,:))]*sqrt(D)

%fprintf(1, '\n L2 error by mapa_K: ')
%[mean(L2Errors_mapa_K(1,:)); mean(L2Errors_mapa_K(2,:))]*sqrt(D)

fprintf(1, '\n REF        mapa        mapa_K\n')

fprintf(1, '\n percentage mean: \n')
squeeze(mean(p,2))

fprintf(1, '\n percentage std: \n')
squeeze(std(p,0,2))

fprintf(1, '\n time: \n')
squeeze(mean(t,2))

fprintf(1, '\n eps \n')
mean(eps,2)*sqrt(D)

dims = int2str(dims);
dims = dims(1:3:end);
eval(['save mapa_only_d' dims 'D' int2str(D) ' p t eps planeDims_mapa L2Errors_mapa'])
