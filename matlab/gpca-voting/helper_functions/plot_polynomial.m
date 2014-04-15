% Render a homogeneous polynomial in two variables, or the variety of a
% homogeneous variable in three variables.  The data samples are used to
% determine the plotting range.
function plot_polynomial(samples, veroneseMapOrder, polynomialCoefficients, aprioriGroupBases)

DEBUG = 2;

[ambientSpaceDimension, sampleCount] = size(samples);

if ambientSpaceDimension ==2,
    
    maxXY = max(samples, [], 2);
    minXY = min(samples, [], 2);
    
    gridSize = 30;
    
    evaluationSamples = zeros(2,gridSize.^2);
    evaluationSampleCount = 1;
    for x = linspace(minXY(1), maxXY(1), gridSize);
        for y = linspace(minXY(2), maxXY(2), gridSize);
            evaluationSamples(:,evaluationSampleCount) = [x; y];
            evaluationSampleCount = evaluationSampleCount + 1;
        end
    end
    xVector = linspace(minXY(1), maxXY(1), gridSize)';
    yVector = linspace(minXY(2), maxXY(2), gridSize)';
    
    % Since this code is not intended to be fast, call
    % generate_veronese_maps to evaluate the polynomial concisely.
    mappedEvaluationData = generate_veronese_maps(evaluationSamples, veroneseMapOrder, 'one');
    zVector = polynomialCoefficients' * mappedEvaluationData;
    zArray = reshape(zVector, [gridSize, gridSize]);
    
    %figure
    %polynomialPlot = surfc(xVector,yVector,zArray);
    polynomialPlot = surf(xVector,yVector,zArray);
    
    axis tight
    %colormap white
    
    %doubleHot = [hot; flipud(hot)]
    colormaps_gpca;  % Load custom colormaps
    colormap(doubleHot);
    %alpha('clear')
    alpha(.5)
    
    %plot_data
    
    %title('The segmentation polynomial as a function of the two dimensional data.')
    
elseif ambientSpaceDimension == 3,
    if(DEBUG >=2),
        disp('Computing data for polynomial zero set...')
    end
    
    % Determine the boundaries of the volume that we care about.
    maxXYZ = max(samples, [], 2);
    minXYZ = min(samples, [], 2);
    gridSize = 50;
    xVector = linspace(minXYZ(1), maxXYZ(1), gridSize)';
    yVector = linspace(minXYZ(2), maxXYZ(2), gridSize)';
    zVector = linspace(minXYZ(3), maxXYZ(3), gridSize)';
    
    % Generate a set of X, Y, Z triplets at which to evaluate the
    % polynomial
    evaluationSamples = zeros(3,gridSize.^3);
    evaluationSampleCount = 1;
    for x = linspace(minXYZ(1), maxXYZ(1), gridSize);
        for y = linspace(minXYZ(2), maxXYZ(2), gridSize);
            for z = linspace(minXYZ(3), maxXYZ(3), gridSize);
                evaluationSamples(:,evaluationSampleCount) = [x y z]';
                evaluationSampleCount = evaluationSampleCount + 1;
            end
        end
    end
    
    %     % Generate another set of random X, Y, Z triples that is randomly
    %     % chosen, and increasingly dense towards the center, to fix the
    %     % glitches at the center of the plot.
    %     extraSampleCount = 1000;
    %     extraEvaluationSamples = randn(extraSampleCount, 3);
    %     scaleFactors = rand(1, extraSampleCount) .^ (1/3);  % Gives uniform sampling inside the sphere.
    %     scaleFactors = scaleFactors .^ 2; % Higher powers concentrate the data closer to zero.
    %     scaleFactors = scaleFactors * min(abs([minXYZ; maxXYZ])); % Rescale the data so that it doesn't stick out too far.
    %     extraEvaluationSamples = diag(sparse(scaleFactors)) * extraEvaluationSamples;
    %     
    %     % Append the new values onto the previously generated data.
    %     evaluationSamples = [evaluationSamples extraEvaluationSamples'];
    %     xVector = [xVector; extraEvaluationSamples(:,1)];
    %     yVector = [yVector; extraEvaluationSamples(:,2)];
    %     zVector = [zVector; extraEvaluationSamples(:,3)];    
    
    % Evaluate the polynomial at the above set of values.
    mappedEvaluationData = generate_veronese_maps(evaluationSamples, veroneseMapOrder, 'one');
    wVector = (polynomialCoefficients' * mappedEvaluationData)';
    wArray = reshape(wVector, [gridSize, gridSize, gridSize]);
    %wArray = permute(wArray, [2 1 3]);
    %wArray = permute(wArray, [1 3 2]);
    wArray = permute(wArray, [2 3 1]);
    %wArray = permute(wArray, [3 2 1]);
    
    % Compute a color map for the entire space based on how close the point
    % is to the various subspaces.  The idea is that the limit planes will
    % be colored the same, and it will likewise be obvious where the lines
    % are.
    
    % Define vectors for the various group colors.
    % colors = 'rbkmgcy'
    groupColors = [1 0 0; 0 1 0; 0 0 0; 1 0 1; 0 1 0; 0 1 1; 1 1 0];
    
    % Compute the distance between each evaluation point and each group.
    
    if(DEBUG >=2),
        disp('Running isosurface...');
    end
    % Plot the zero set of the polynomial.
    fv = isosurface(xVector,yVector,zVector,wArray,0);
    if(DEBUG >=2),
        disp('Running reducepatch...');
    end
    nfv = reducepatch(fv, .2);  % Reduce the number of patches to speed rendering, while preserving shape.
    p = patch(nfv);
    isonormals(xVector,yVector,zVector,wArray,p)
    set(p,'FaceColor','blue','EdgeColor','black')
    set(p,'EdgeAlpha', .2)
    %set(p, 'FaceColor','white', 'EdgeColor', 'black')
    view([-65,20])
    axis vis3d
    camlight right; 
    %set(gcf,'Renderer','zbuffer'); 
    set(gcf, 'Renderer', 'OpenGL');
    %    lighting phong
    lighting gouraud
    
    alpha(.5)
    
%     disp('Hit the any key to stop the animation.')
%     dTheta = 3;  % Degrees increments.
    %   set(gcf, 'CurrentCharacter', []);
%     while(get(gcf, 'CurrentCharacter')~='x');
%         camorbit(dTheta,0,'camera');
%         drawnow;
%     end
    
end
