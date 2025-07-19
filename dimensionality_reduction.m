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
combinations(2).eventName = 'respTimes';
combinations(2).selectedEvents = respTimes;

combinations(3).areaID = 10; % VISp
combinations(3).eventName = 'goTimes';
combinations(3).selectedEvents = goTimes;

combinations(4).areaID = 1; % VISp
combinations(4).eventName = 'stimTimes';
combinations(4).selectedEvents = stimTimes;

combinations(5).areaID = 1; % VISp
combinations(5).eventName = 'respTimes';
combinations(5).selectedEvents = respTimes;

combinations(6).areaID = 1; % VISp
combinations(6).eventName = 'goTimes';
combinations(6).selectedEvents = goTimes;

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
    numTrialsPerNeuron = size(spikes(1).binnedArray, 1); % Re-confirm as it might vary if 'spikes' differs

    % current_timeByspikes = zeros(current_nClusters * numTrialsPerNeuron, current_numTimeBins);
    current_timeByspikes = zeros(current_nClusters, current_numTimeBins);

    for idx = 1:current_nClusters
        currentBinnedArray = spikes(idx).binnedArray / binSize; % Assuming conversion to firing rate
        startRow = (idx - 1) * numTrialsPerNeuron + 1;
        endRow = idx * numTrialsPerNeuron;
        %current_timeByspikes(startRow:endRow, :) = currentBinnedArray;
        current_timeByspikes(idx, :) = mean(currentBinnedArray,1);
    end
    analysisResults(comboIdx).timeByspikes = current_timeByspikes;

    % Perform PCA for current combination
    [coeff, score, latent, tsquared, explained, mu] = pca( sqrt(current_timeByspikes'));

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

    % % Plot cumulative explained variance
    % plot(1:length(cumulativeExplained), cumulativeExplained, 'rx-', ...
    %      'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Cumulative Explained Variance');
    % hold off;

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



