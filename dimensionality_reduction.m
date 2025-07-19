%% Multivariate Analysis of Steinmetz neuron spikes by various regions 
% Run first: openSession or first two blocks of exploreSteinmetz
clear all; close all; clc
sesPath = '../data/Steinmetz/Cori_2016-12-18'; % sample with both motor and sensory areas
addpath(genpath('../packages/spikes-master'))

%% Read in spike data .. ~5 sec and construct some convenience variables
% Note that regions are indexed common style from 1 to regions.N, but neurons are indexed Python-style from 0 to neurons.N-1
[S, regions, neurons, trials] = stOpenSession(sesPath);  % load .npy files in which data stored

%% Book-keeping
sessionTime = S.spikes.times(end); % total time, assuming start at 0
stimTimes = trials.visStimTime; 
respTimes = trials.responseTime;
goTimes = S.trials.goCue_times;
% construct logical variable for spike timestamps in trials
inTrial = false(size(S.spikes.times,1),1);
for kk = 1:trials.N
    inTrial( S.spikes.times > stimTimes(kk) & S.spikes.times < respTimes(kk) ) = true;
end

% Some useful book-keeping variables used frequently later
neuronNumEdges = (0:neurons.N) - 0.5; % bin edges for using histcounts to count neuron cluster IDs in spikes


%% Define the combinations
combinations(1).areaID = 10; % VISp
combinations(1).eventName = 'stimTimes';
combinations(1).selectedEvents = stimTimes;

combinations(2).areaID = 10; % VISp
combinations(2).eventName = 'goTimes';
combinations(2).selectedEvents = goTimes;

combinations(3).areaID = 10; % VISp
combinations(3).eventName = 'respTimes';
combinations(3).selectedEvents = respTimes;

combinations(4).areaID = 1; % CA1
combinations(4).eventName = 'stimTimes';
combinations(4).selectedEvents = stimTimes;

combinations(5).areaID = 1; % CA1
combinations(5).eventName = 'goTimes';
combinations(5).selectedEvents = goTimes;

combinations(6).areaID = 1; % CA1
combinations(6).eventName = 'respTimes';
combinations(6).selectedEvents = respTimes;

areaIdToNameMap = containers.Map('KeyType', 'double', 'ValueType', 'char');
areaIdToNameMap(10) = 'VISp'; % Visual Primary Cortex
areaIdToNameMap(1) = 'CA1';  % Hippocampus CA1 region

%% PSTH and PCA
win = [0,0.5];
binSize = 0.02;

% Initialize a struct array to store results for each combination
analysisResults = struct('areaID', {}, 'eventName', {}, 'params', {}, ...
                         'timeBinCenters', {}, 'spikesProcessed', {}, ...
                         'timeByspikes', {}, 'pcaResults', {});

for comboIdx = 1:length(combinations)
    current_areaID = combinations(comboIdx).areaID;
    current_eventName = combinations(comboIdx).eventName;
    current_selectedEvents = combinations(comboIdx).selectedEvents;
    
    % Store current combination's details
    analysisResults(comboIdx).areaID = current_areaID;
    analysisResults(comboIdx).eventName = current_eventName;
    analysisResults(comboIdx).params.win = win;
    analysisResults(comboIdx).params.binSize = binSize;

    % PSTH (align to visual stimulus) for current combination
    clear spikes;
    spikes = struct();

    current_clusterIDs = find(neurons.region == current_areaID);
    current_nClusters = length(current_clusterIDs);

    if current_nClusters == 0
        warning('No neurons found for AreaID %d. Skipping this combination.', current_areaID);
        continue;
    end

    bins_for_current_combo = []; % To capture 'bins' output from psthAndBA

    for idx = 1:current_nClusters
        spikes(idx).clu = (current_clusterIDs(idx));
        spikes(idx).spikeIndex = find(S.spikes.clusters(:) == spikes(idx).clu);
        spikes(idx).spiketimes = S.spikes.times(spikes(idx).spikeIndex);

        [spikes(idx).psth, bins_for_current_combo, spikes(idx).rasterX, spikes(idx).rasterY, ...
         spikes(idx).spikeCounts, spikes(idx).binnedArray] = psthAndBA(spikes(idx).spiketimes, current_selectedEvents, win, binSize);
    end

    analysisResults(comboIdx).spikesProcessed = spikes;

    current_numTimeBins = size(spikes(1).binnedArray, 2);
    analysisResults(comboIdx).timeBinCenters = linspace(win(1) + binSize/2, win(2) - binSize/2, current_numTimeBins);

    % Prepare data for PCA for current combination
    currentFR = zeros(current_nClusters, current_numTimeBins);

    for idx = 1:current_nClusters
        currentFR(idx, :) = spikes(idx).psth;
    end
    analysisResults(comboIdx).timeByspikes = currentFR;

    % Perform PCA for current combination
    [coeff, score, latent, ~, explained, mu] = pca( sqrt(currentFR'));

    analysisResults(comboIdx).pcaResults.coeff = coeff;
    analysisResults(comboIdx).pcaResults.score = score;
    analysisResults(comboIdx).pcaResults.latent = latent;
    analysisResults(comboIdx).pcaResults.explained = explained;
    analysisResults(comboIdx).pcaResults.mu = mu;

    fprintf('PCA Results for Combination %d (Area %d, Event: %s):\n', ...
            comboIdx, current_areaID, current_eventName);
    fprintf('  Total variance explained by first 5 PCs: %.2f%%\n', sum(explained(1:5)));
    fprintf('%%\n');
end

  
%% Visualization 1: Explained Variance (Scree Plots for all Combinations)

figure('Position', [100, 100, 1200, 800]); % Create a larger figure window for 6 subplots
sgtitle('Scree Plots: Explained Variance by Principal Components Across Conditions', 'FontSize', 16, 'FontWeight', 'bold');

% Loop through each combination to create a subplot
for comboIdx = 1:length(analysisResults)
    % Select the current subplot (2 rows, 3 columns)
    subplot(2, 3, comboIdx); % Adjust subplot layout if you have more/fewer than 6 combinations

    % Retrieve data for the current combination
    current_explained = analysisResults(comboIdx).pcaResults.explained;
    current_areaID = analysisResults(comboIdx).areaID;
    current_eventName = analysisResults(comboIdx).eventName;

    % --- Get the anatomical name for the current areaID ---
    if isKey(areaIdToNameMap, current_areaID)
        areaName = areaIdToNameMap(current_areaID);
    else
        areaName = sprintf('ID %d (Unknown Area)', current_areaID); % Fallback for IDs not in the map
    end

    % Calculate cumulative explained variance
    cumulativeExplained = cumsum(current_explained);

    % Plot individual PC variance
    plot(1:length(current_explained), current_explained, 'o-', ...
         'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Individual PC Variance');
    hold on;

    % Set plot properties
    xlabel('Principal Component Number', 'FontSize', 12);
    ylabel('Variance Explained (%)', 'FontSize', 12);
    % Update the title to use the anatomical name
    title(sprintf('%s, Event: %s', areaName, current_eventName), 'FontSize', 14); % Improved title
    grid on;
    %legend('Location', 'best', 'FontSize', 10);

    % Optional: Adjust y-axis limits if needed for consistency across plots
    ylim([0, max(max(current_explained), 45)]);

end

%% Visualization 2

figure('Position', [100, 100, 1200, 800]); % Create a larger figure window for 6 subplots
sgtitle('Scree Plots: Explained Variance by Principal Components Across Conditions', 'FontSize', 16, 'FontWeight', 'bold');

% Loop through each combination to create a subplot
for comboIdx = 1:length(analysisResults)
    % Select the current subplot (2 rows, 3 columns)
    subplot(2, 3, comboIdx); % Adjust subplot layout if you have more/fewer than 6 combinations

    % Retrieve data for the current combination
    current_explained = analysisResults(comboIdx).pcaResults.explained;
    current_areaID = analysisResults(comboIdx).areaID;
    current_eventName = analysisResults(comboIdx).eventName;

    % --- Get the anatomical name for the current areaID ---
    if isKey(areaIdToNameMap, current_areaID)
        areaName = areaIdToNameMap(current_areaID);
    else
        areaName = sprintf('ID %d (Unknown Area)', current_areaID); % Fallback for IDs not in the map
    end

    % Calculate cumulative explained variance
    cumulativeExplained = cumsum(current_explained);

    % Plot cumulative explained variance
    plot(1:length(cumulativeExplained), cumulativeExplained, 'rx-', ...
         'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Cumulative Explained Variance');
    hold off;

    % Set plot properties
    xlabel('Principal Component Number', 'FontSize', 12);
    ylabel('Variance Explained (%)', 'FontSize', 12);
    % Update the title to use the anatomical name
    title(sprintf('%s, Event: %s', areaName, current_eventName), 'FontSize', 14); % Improved title
    grid on;
    %legend('Location', 'best', 'FontSize', 10);

    % Optional: Adjust y-axis limits if needed for consistency across plots
    ylim([10, max(max(current_explained), 100)]);

end


%% Visualization 3: Projecting Data onto Principal Components (3D Scores Plots for all Combinations)

figure('Position', [100, 100, 1500, 900]); % Create a larger figure window for 3D subplots
sgtitle('PCA Scores: Projection of Time Bins onto PC1, PC2, PC3 Across Conditions', ...
        'FontSize', 16, 'FontWeight', 'bold');

% Loop through each combination to create a subplot
for comboIdx = 1:length(analysisResults)
    % Select the current subplot (2 rows, 3 columns)
    subplot(2, 3, comboIdx); % Adjust subplot layout (e.g., 3,2,comboIdx if preferred)

    % Retrieve data for the current combination
    current_score = analysisResults(comboIdx).pcaResults.score;
    current_timeBinCenters = analysisResults(comboIdx).timeBinCenters;
    current_areaID = analysisResults(comboIdx).areaID;
    current_eventName = analysisResults(comboIdx).eventName;

    % --- Get the anatomical name for the current areaID ---
    if isKey(areaIdToNameMap, current_areaID)
        areaName = areaIdToNameMap(current_areaID);
    else
        areaName = sprintf('ID %d (Unknown Area)', current_areaID); % Fallback
    end

    % Check if score has at least 3 components to plot in 3D
    if size(current_score, 2) < 3
        warning('Combination %d (Area %s, Event %s): Not enough PCs (%d) for 3D plot. Skipping.', ...
                comboIdx, areaName, current_eventName, size(current_score, 2));
        title(sprintf('%s, Event: %s (Not enough PCs for 3D)', areaName, current_eventName), 'FontSize', 12);
        grid on;
        continue; % Skip to the next iteration
    end

    % Plotting the 3D scores, color-coded by time progression
    scatter3(current_score(:,1), current_score(:,2), current_score(:,3), ...
             50, current_timeBinCenters, 'filled', 'MarkerFaceAlpha', 0.7); % Marker size 50, alpha for transparency
    
    % Add labels and title
    xlabel('PC1 Score', 'FontSize', 10); % Slightly smaller font for subplots
    ylabel('PC2 Score', 'FontSize', 10);
    zlabel('PC3 Score', 'FontSize', 10);
    title(sprintf('%s, Event: %s', areaName, current_eventName), 'FontSize', 12); % Subplot title

    grid on;
    colormap(gca, jet); % Apply colormap to current axes
    
    % Add a color bar for the time progression
    if comboIdx == length(analysisResults) % Only add colorbar once for the last subplot, or more strategically
        cb = colorbar('Location', 'eastoutside');
        cb.Label.String = 'Time (s)';
        cb.FontSize = 10;
    end
    
    % Optional: Adjust view angle for better 3D perception
    view(3); % Default 3D view
    view(45, 30); % Example: azimuth 45, elevation 30
end


%% NMF

% Define the number of NMF components (k)
% This is a crucial parameter for NMF. You can adjust this.
numNMFComponents = 5;

% Loop through each combination in the already computed analysisResults
for comboIdx = 1:length(analysisResults)
    current_areaID = analysisResults(comboIdx).areaID;
    current_eventName = analysisResults(comboIdx).eventName;
    current_numTimeBins = size(analysisResults(comboIdx).timeByspikes, 2); % Get num time bins for this combo
    current_nClusters = size(analysisResults(comboIdx).timeByspikes, 1); % Get num neurons for this combo

    % Retrieve the pre-computed data matrix for factorization
    % This corresponds to the 'currentFR_rates' from the previous code,
    % which was (Neurons x TimeBins).
    currentFR_rates = analysisResults(comboIdx).timeByspikes;

 
    nmf_input_matrix = sqrt(currentFR_rates'); % Dimensions: numTimeBins x current_nClusters

    % Ensure data is explicitly non-negative for NMF
    if any(nmf_input_matrix(:) < 0)
        error('NMF Input Error: Data for Combination %d (Area %d, Event: %s) contains negative values after sqrt. NMF requires non-negative input.', ...
              comboIdx, current_areaID, current_eventName);
    end

    % Perform NMF using 'nnmf'
    % W: Basis matrix (numTimeBins x numNMFComponents) -> Temporal patterns (columns)
    % H: Coefficient matrix (numNMFComponents x current_nClusters) -> Neuron activations (rows)
    % D: Divergence (reconstruction error)
    [W_nmf, H_nmf, D_nmf] = nnmf(nmf_input_matrix, numNMFComponents, 'alg', 'mult');

    % Store NMF results within the analysisResults struct
    analysisResults(comboIdx).nmfResults.W = W_nmf; % Temporal patterns are here (columns)
    analysisResults(comboIdx).nmfResults.H = H_nmf; % Neuron activations/coefficients (rows)
    analysisResults(comboIdx).nmfResults.D = D_nmf; % Divergence (reconstruction error)
    analysisResults(comboIdx).nmfResults.numComponents = numNMFComponents;

    fprintf('  NMF performed. Reconstruction divergence: %.4f\n', D_nmf);
end


%% Visualization of NMF Temporal Patterns (Basis Vectors - W)
% Plotting the columns of the W matrix for each combination.
% W has dimensions: numTimeBins x numNMFComponents

figure('Position', [100, 100, 1500, 900]); % Create a large figure for 6 subplots
sgtitle('NMF Temporal Basis Vectors (W) Across Conditions', ...
        'FontSize', 16, 'FontWeight', 'bold');

for comboIdx = 1:length(analysisResults)
    subplot(2, 3, comboIdx); % Adjust subplot layout as needed

    % Retrieve NMF data for the current combination
    current_W = analysisResults(comboIdx).nmfResults.W;
    current_numNMFComponents = analysisResults(comboIdx).nmfResults.numComponents;
    current_timeBinCenters = analysisResults(comboIdx).timeBinCenters;
    current_areaID = analysisResults(comboIdx).areaID;
    current_eventName = analysisResults(comboIdx).eventName;

    % Get anatomical name
    if isKey(areaIdToNameMap, current_areaID)
        areaName = areaIdToNameMap(current_areaID);
    else
        areaName = sprintf('ID %d (Unknown Area)', current_areaID);
    end

    % Check if NMF was successfully performed and has components
    if isempty(current_W) || size(current_W, 2) < 1
        warning('Combination %d (Area %s, Event %s): NMF results not found or no components.', ...
                comboIdx, areaName, current_eventName);
        title(sprintf('%s, Event: %s (NMF: No Data)', areaName, current_eventName), 'FontSize', 12);
        grid on;
        continue;
    end

    % Plot each NMF component (column of W)
    colors = lines(current_numNMFComponents); % Generate distinct colors for each component
    hold on;
    for k = 1:current_numNMFComponents
        if k <= size(current_W, 2) % Ensure k is within actual number of columns in W
            plot(current_timeBinCenters, current_W(:, k), 'Color', colors(k,:), ...
                 'LineWidth', 1.5, 'DisplayName', sprintf('NMF Comp %d', k));
        end
    end
    hold off;

    xlabel('Time (s)', 'FontSize', 10);
    ylabel('Factor Value (a.u.)', 'FontSize', 10); % Arbitrary units
    title(sprintf('%s, Event: %s', areaName, current_eventName), 'FontSize', 12);
    grid on;
    legend('Location', 'best', 'FontSize', 8);
    
    % Ensure y-axis starts at zero for non-negative factors
    ylim_current = ylim;
    ylim([0, ylim_current(2)]);
end


%% Visualization of NMF Neuron-Trial Activations (H Matrix)
% --- Visualization of NMF Neuron Activations (H Matrix - Per Neuron) ---
% Now, each point in the scatter plot represents a SINGLE NEURON (average activity)
% H has dimensions: numNMFComponents x nClusters

figure('Position', [100, 100, 1500, 900]);
sgtitle('NMF Neuron Activations (H) - Factor 1 vs Factor 2 (Per Neuron)', ...
        'FontSize', 16, 'FontWeight', 'bold');

for comboIdx = 1:length(analysisResults)
    subplot(2, 3, comboIdx);

    current_H = analysisResults(comboIdx).nmfResults.H;
    current_areaID = analysisResults(comboIdx).areaID;
    areaName = getAreaName(areaIdToNameMap, current_areaID); % Helper function below
    current_eventName = analysisResults(comboIdx).eventName;
    
    % Get the number of neurons (clusters) for the current combination.
    % This is the number of columns in the H matrix.
    current_nClusters = size(analysisResults(comboIdx).timeByspikes, 1); 

    if isempty(current_H) || size(current_H, 1) < 2
        warning('Combination %d (Area %s, Event %s): NMF results not found or <2 components for H plot.', ...
                comboIdx, areaName, current_eventName);
        title(sprintf('%s, Event: %s (NMF: Not enough factors for 2D plot)', areaName, current_eventName), 'FontSize', 12);
        grid on;
        continue;
    end

    % Scatter plot of Factor 1 activation vs Factor 2 activation
    % Each point now represents a SINGLE NEURON.
    % H(1,:) contains activations for Factor 1 across all neurons.
    % H(2,:) contains activations for Factor 2 across all neurons.

    % Generate colors for each neuron
    colors_for_neurons = parula(current_nClusters);
    hold on;
    for neuron_idx = 1:current_nClusters
        % Plotting one point per neuron using its Factor 1 and Factor 2 activation
        scatter(current_H(1, neuron_idx), ... % Factor 1 activation for this neuron
                current_H(2, neuron_idx), ... % Factor 2 activation for this neuron
                50, colors_for_neurons(neuron_idx,:), 'filled', 'MarkerFaceAlpha', 0.7); % Increased size for individual points
    end
    hold off;

    xlabel('NMF Factor 1 Activation', 'FontSize', 10);
    ylabel('NMF Factor 2 Activation', 'FontSize', 10);
    title(sprintf('%s, Event: %s', areaName, current_eventName), 'FontSize', 12);
    grid on;
    % Ensure axes start at 0 for non-negative activations
    xlim_current = xlim;
    ylim_current = ylim;
    xlim([0, xlim_current(2)]);
    ylim([0, ylim_current(2)]);
end


% --- Helper function (place this at the end of your script or in a separate .m file) ---
function name = getAreaName(map, id)
    if isKey(map, id)
        name = map(id);
    else
        name = sprintf('ID %d (Unknown Area)', id);
    end
end


%% Compute and Store Reconstruction Errors for PCA and NMF

% Loop through each combination in the 'analysisResults' struct array
for comboIdx = 1:length(analysisResults)
    current_areaID = analysisResults(comboIdx).areaID;
    current_eventName = analysisResults(comboIdx).eventName;
    
    fprintf('  Processing Combination %d: Area %d, Event: %s\n', ...
            comboIdx, current_areaID, current_eventName);

    % --- Retrieve Original Data Matrix for Factorization ---
    % This is the input that was fed into PCA and NMF.
    % It was defined as sqrt(currentFR_rates'), where currentFR_rates was timeByspikes.
    % So, retrieve analysisResults(comboIdx).timeByspikes and transform it back.
    currentFR_rates = analysisResults(comboIdx).timeByspikes; % Dimensions: nClusters x numTimeBins
    original_data_for_factorization = sqrt(currentFR_rates'); % Dimensions: numTimeBins x nClusters

    % --- PCA Reconstruction Error ---
    if isfield(analysisResults(comboIdx).pcaResults, 'score') && ...
       isfield(analysisResults(comboIdx).pcaResults, 'coeff') && ...
       isfield(analysisResults(comboIdx).pcaResults, 'mu')
        
        % Retrieve the FULL score, coeff, and mu from analysisResults
        current_score_full = analysisResults(comboIdx).pcaResults.score;
        current_coeff_full = analysisResults(comboIdx).pcaResults.coeff;
        current_mu = analysisResults(comboIdx).pcaResults.mu;

        if isfield(analysisResults(comboIdx).nmfResults, 'numComponents')
            numComponentsForReconstruction = analysisResults(comboIdx).nmfResults.numComponents;
        else
            numComponentsForReconstruction = 5; % Default to 5 PCs for comparison
            warning('NMF components (k) not found for combo %d. Defaulting to %d PCs for comparison.', comboIdx, numComponentsForReconstruction);
        end

        
        % Reconstruct the data using *only the first 'numComponentsForReconstruction' COLUMNS*
        % using the retrieved full matrices (current_score_full, current_coeff_full)
        pca_reconstructed_data = (current_score_full(:, 1:numComponentsForReconstruction) * ...
                                  current_coeff_full(:, 1:numComponentsForReconstruction)') + current_mu;
        
        % Calculate Frobenius norm of the error: ||Original - Reconstructed||_F
        pca_reconstruction_error_fro = norm(original_data_for_factorization - pca_reconstructed_data, 'fro');
        analysisResults(comboIdx).pcaResults.reconstructionErrorFro = pca_reconstruction_error_fro;
        
        fprintf('    PCA Frobenius Reconstruction Error (using %d PCs): %.4f\n', ...
                numComponentsForReconstruction, pca_reconstruction_error_fro);
    else
        fprintf('    PCA results (score, coeff, mu) not found for this combination. Skipping PCA error calculation.\n');
    end


    % --- NMF Reconstruction Error ---
    if isfield(analysisResults(comboIdx).nmfResults, 'W') && ...
       isfield(analysisResults(comboIdx).nmfResults, 'H')
        
        current_W_nmf = analysisResults(comboIdx).nmfResults.W;
        current_H_nmf = analysisResults(comboIdx).nmfResults.H;

        % Reconstruct the data using NMF components
        nmf_reconstructed_data = current_W_nmf * current_H_nmf;
        
        % Calculate Frobenius norm of the error: ||Original - Reconstructed||_F
        nmf_reconstruction_error_fro = norm(original_data_for_factorization - nmf_reconstructed_data, 'fro');
        analysisResults(comboIdx).nmfResults.reconstructionErrorFro = nmf_reconstruction_error_fro;
        
        % Also display the intrinsic NMF divergence if available
        if isfield(analysisResults(comboIdx).nmfResults, 'D')
            fprintf('    NMF Reconstruction Divergence (KL): %.4f\n', analysisResults(comboIdx).nmfResults.D);
        end
        fprintf('    NMF Frobenius Reconstruction Error: %.4f\n', nmf_reconstruction_error_fro);
    else
        fprintf('    NMF results (W, H) not found for this combination. Skipping NMF error calculation.\n');
    end
    fprintf('%%\n');
end



%% UMAP

% --- UMAP Parameters ---
% Using parameters from your run_umap example. Adjust these as needed.
numUMAPDimensions = 3; % 
numUMAPNeighbors = 50; 
umapVerboseSetting = 'none';

% Loop through each combination in the 'analysisResults' struct array
for comboIdx = 1:length(analysisResults)
    current_areaID = analysisResults(comboIdx).areaID;
    current_eventName = analysisResults(comboIdx).eventName;
    
    if isempty(analysisResults(comboIdx).timeByspikes)
        fprintf('    Skipping UMAP for this combo: timeByspikes data is empty.\n');
        % Ensure umapResults is initialized even if skipped, to prevent errors later.
        analysisResults(comboIdx).umapResults = struct(); 
        continue; 
    end
    currentFR_rates = analysisResults(comboIdx).timeByspikes; % Dimensions: nClusters x numTimeBins

    % Apply the 'second squared root' as specifically requested for UMAP input
    dataForUMAP_input = sqrt(currentFR_rates); % Dimensions: numTimeBins x nClusters

    % Get the actual number of observations (time bins) and features (neurons) for UMAP input
    [numObservationsUMAP, numFeaturesUMAP] = size(dataForUMAP_input);

    % --- Perform UMAP for current combination ---
    % UMAP requires num_observations >= n_neighbors + 1
    if numObservationsUMAP >= (numUMAPNeighbors + 1)
        try
            % Call your specific run_umap function
            % Input `dataForUMAP_input` is (Observations x Features), so (Time Bins x Neurons)
            [umap_reduction, umap_model, clusterIdentifiers, umap_extras] = run_umap(double(dataForUMAP_input), ...
                                                                'n_components', numUMAPDimensions, ...
                                                                'n_neighbors', numUMAPNeighbors, ...
                                                                'verbose', umapVerboseSetting);

            % Store UMAP results in analysisResults
            analysisResults(comboIdx).umapResults.embedding = umap_reduction;
            analysisResults(comboIdx).umapResults.model = umap_model; % The trained UMAP object
            analysisResults(comboIdx).umapResults.clusterIdentifiers = clusterIdentifiers; % DBSCAN/DBM cluster IDs
            analysisResults(comboIdx).umapResults.extras = umap_extras; % Additional results
            analysisResults(comboIdx).umapResults.params.numDimensions = numUMAPDimensions;
            analysisResults(comboIdx).umapResults.params.numNeighbors = numUMAPNeighbors;
            analysisResults(comboIdx).umapResults.params.minDist = 0.1; % Add a common minDist default if not in run_umap

            fprintf('    UMAP embedding computed (%d time bins in %dD).\n', ...
                    size(umap_reduction, 1), size(umap_reduction, 2));
        catch umap_err
            warning('UMAP Error for Combination %d (Area %d, Event: %s): %s. Skipping UMAP for this combo.', ...
                    comboIdx, current_areaID, current_eventName, umap_err.message);
            analysisResults(comboIdx).umapResults = struct(); % Store empty if error
        end
    else
        fprintf('    UMAP skipped for Combination %d (Area %d, Event: %s): Not enough observations (%d) for %d neighbors (need at least %d).\n', ...
                comboIdx, current_areaID, current_eventName, numObservationsUMAP, numUMAPNeighbors, numUMAPNeighbors + 1);
        analysisResults(comboIdx).umapResults = struct(); % Store empty UMAP
    end
    fprintf('%%\n');
end



%% if export to python
% Define the full path where you want to save your .mat file
% It's good practice to use fullfile for cross-platform compatibility
output_directory = '/Users/yangy1/Desktop/NDS_2025_project/project_scripts'; % e.g., create a subfolder
output_filename = 'psth_for_python_viz.mat';
full_export_path = fullfile(output_directory, output_filename);

% Call the function
exported_psth_data = exportPSTHDataForPython(spikes, win, binSize, full_export_path);

% You can now load 'full_export_path' in Python using scipy.io.loadmat.




%% PSTH (align to visual stimulus)
areaID = 10; % for area VISp
win = [-0.5,0.5]; % Original window, -0.5s to +0.5s around stimTime
binSize = 0.02;
selectedEvents = stimTimes;
clusterIDs = find(neurons.region==areaID);
nClusters = length(clusterIDs);

% Initialize 'bins' outside the loop if it's consistent for all neurons
% (which it should be if win and binSize are fixed)
% This way, `timeBinCenters` can be calculated once after the loop.
bins = []; % Initialize it so it's guaranteed to exist for the calculation below

for idx = 1:nClusters
    spikes(idx).clu = (clusterIDs(idx));
    spikes(idx).spikeIndex = find(S.spikes.clusters(:)==spikes(idx).clu);
    spikes(idx).spiketimes = S.spikes.times(spikes(idx).spikeIndex);

    %organize into matrix around stimTime:
    [spikes(idx).psth, bins, spikes(idx).rasterX, spikes(idx).rasterY, spikes(idx).spikeCounts, spikes(idx).binnedArray] = psthAndBA(spikes(idx).spiketimes, selectedEvents, win, binSize);
    % Note: `bins` here will be updated in each iteration,
    % taking the value from the last `idx` iteration. This is generally fine
    % if `win` and `binSize` are constant, as the bin edges will be the same.
end



%% Prepare data for PCA

nClusters = length(spikes); 
numTrialsPerNeuron = size(spikes(1).binnedArray, 1); % Get number of trials (228)
numTimeBins = size(spikes(1).binnedArray, 2); % Get number of time bins (50)

% The total number of rows will be (number of neurons * number of trials per neuron)
% The number of columns will be the number of time bins.
timeByspikes = zeros(nClusters * numTrialsPerNeuron, numTimeBins);

% Loop through each neuron and concatenate its binnedArray
for idx = 1:nClusters
    % Get the binnedArray for the current neuron
    currentBinnedArray = spikes(idx).binnedArray/binSize;

    % Calculate the starting and ending row indices for placing this neuron's data
    startRow = (idx - 1) * numTrialsPerNeuron + 1;
    endRow = idx * numTrialsPerNeuron;

    % Concatenate (vertically stack) the current neuron's binnedArray
    timeByspikes(startRow:endRow, :) = currentBinnedArray;
end

[coeff, score, latent, tsquared, explained, mu] = pca( sqrt(timeByspikes'));
disp('PCA Results:');
disp(['Total variance explained by first 5 PCs: ', num2str(sum(explained(1:5))), '%']);



