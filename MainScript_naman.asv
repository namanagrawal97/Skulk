%% Multivariate Analysis of Steinmetz neuron spikes by various regions 
% Run first: openSession or first two blocks of exploreSteinmetz
clear all; close all; clc
sesPath = '../data/Steinmetz/Hench_2017-06-17'; % sample with both motor and sensory areas

% Read in spike data .. ~5 sec and construct some convenience variables
% Note that regions are indexed common style from 1 to regions.N, but neurons are indexed Python-style from 0 to neurons.N-1
[S, regions, neurons, trials] = stOpenSession(sesPath);  % load .npy files in which data stored

% Book-keeping
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

% Parameters to convert between frame numbers for behavior and time in seconds
% The generic way to select frame for X corresponding to a time T is: myFrame = round(( T - XInt)/XSlope
% The generic way to assign a time corresponding to a frame is: myT = XInt + XSlope * frame
% These are mostly to derive time ranges from frame ranges; one must count the numbers of frames
faceframe2timeInt = S.face.timestamps(1,2); % usually 10 - 20 sec after recording start
faceframe2timeSlope = (S.face.timestamps(2,2)-S.face.timestamps(1,2))/(S.face.timestamps(2,1)-S.face.timestamps(1,1)); % usually ~ 1/40Hz
% Note: DeepLabCut variables are on same time as face camera 
% Note: sometimes eye timestamps are fully provided
if size(S.eye.timestamps) == [2 2]
    eyeframe2timeInt = S.eye.timestamps(1,2); % usually 10 - 20 sec after recording start
    eyeframe2timeSlope = (S.eye.timestamps(2,2)-S.eye.timestamps(1,2))/(S.eye.timestamps(2,1)-S.eye.timestamps(1,1)); % usually ~ 1/100Hz
end
% NB Need to filter the eye data
wheelframe2timeInt = S.wheel.timestamps(1,2); % usually 10 - 20 sec after recording start
wheelframe2timeSlope = (S.wheel.timestamps(2,2)-S.wheel.timestamps(1,2))/(S.wheel.timestamps(2,1)-S.wheel.timestamps(1,1)); % ~ 1/2500Hz
%%
% create structure
areaID = 11; % for area VISp
win = [-0.5,0.5];
binSize = 0.02;
selectedEvents = stimTimes; %Visual Stimulus ON
clusterIDs = find(neurons.region==areaID);
nClusters = length(clusterIDs);
for idx = 1:nClusters
    spikes(idx).clu = (clusterIDs(idx));
    spikes(idx).spikeIndex = find(S.spikes.clusters(:)==spikes(idx).clu);
    spikes(idx).spiketimes = S.spikes.times(spikes(idx).spikeIndex);
    
    %organize into matrix around stimTime:
    [spikes(idx).psth, bins, spikes(idx).rasterX, spikes(idx).rasterY, spikes(idx).spikeCounts, spikes(idx).binnedArray] = psthAndBA(spikes(idx).spiketimes, selectedEvents, win, binSize);
end
unique_trials_ids = unique(trials.contrast);
trialTypes = size(unique_trials_ids,1);

%% plot cells
plotCells = 0; % set this to 1, and it will plot the first 20 cells so you can take a look
if isequal(plotCells,1)
sm =  0.25;
colors = turbo(100);
for idx = 1:20
    figure; tiledlayout(2,1); nexttile; axR = gca; nexttile; axP = gca;
    rasterAndPSTHbyCond(spikes(idx).spiketimes, stimTimes, trials.contrast, win, sm, colors, axR, axP)
end
end
%% Plotting neurons by stimulus types
go_trials_idx = find(trials.isStimulus==1);
left_trial_only = find(trials.contrast==-1);

neuron_id = 100; % Choose a neuron ID
spikes_index = find([spikes.clu] == neuron_id);
neuron_spikes = spikes(spikes_index).binnedArray;
figure; tiledlayout(1,9)
for idx = 1:9
    trial_type = unique_trials_ids(idx);
    trial_only = find((trials.contrast==trial_type) & (trials.isStimulus==1));
    neuron_spikes_trial_only = neuron_spikes(trial_only,:);
    ax=nexttile;
    rasterplot_naman(ax,neuron_spikes_trial_only, win, binSize);
end
%% I want to make a tuning heatmap
% Loop through each neuron
averageFiringRates = zeros(nClusters, trialTypes);

for neuronID = 1:nClusters
    disp(neuronID)
    neuron_spikes = spikes(neuronID).binnedArray;
    % Loop through each trial type
    for trialidx = 1:trialTypes
        disp(trialidx)
        trialType=unique_trials_ids(trialidx);
        trial_only = find((trials.contrast==trialType) & (trials.isStimulus==1));
        spikeMatrix = neuron_spikes(trial_only,:);        
        % Calculate the average firing rate using our function
        averageFiringRates(neuronID, trialidx) = calculateAverageFiring(spikeMatrix);
    end
end
%%
% --- 4. Filter out Low-Firing Neurons ---
% Define a threshold (in Hz) for the peak firing rate.
% Neurons whose maximum firing rate across all conditions is below this
% threshold will be removed.
firingRateThreshold = 5.0; % e.g., 1 spike/sec


% Find the original IDs of neurons that meet the threshold.
neuronsToKeep = find(max(averageFiringRates, [], 2) >= firingRateThreshold);

% Create a new matrix with only the high-firing neurons.
filteredFiringRates = averageFiringRates(neuronsToKeep, :);
%%
% --- 5. Sort and Plot FILTERED Neurons ---
if ~isempty(filteredFiringRates)
    % Find the preferred trial for the remaining neurons
    [filteredMaxRates, filteredPrefIndex] = max(filteredFiringRates, [], 2);
    
    % Create sorting criteria for the filtered set
    filteredSortCriteria = [filteredPrefIndex, filteredMaxRates];
    
    % Get the new sorted order (indices relative to the filtered set)
    [~, sortedFilteredOrder] = sortrows(filteredSortCriteria, [1, -2]);
    
    % Reorder the filtered firing rate matrix
    finalSortedRates = filteredFiringRates(sortedFilteredOrder, :);
    
    % Get the original neuron IDs in their final sorted order for labeling
    finalSortedYLabels = string(neuronsToKeep(sortedFilteredOrder));
  
    % --- 3. Generate the Heatmap ---
    figure; % Create a new figure for the heatmap
    
    h = heatmap(unique_trials_ids, 1:length(neuronsToKeep), finalSortedRates);
    h.YDisplayLabels = finalSortedYLabels;
    
    h.Title = 'Average Neuronal Firing Rate (0-1s) Across Conditions';
    h.XLabel = 'Trial Type';
    h.YLabel = 'Neuron ID';
    h.Colormap = jet; % Use a vibrant colormap
    
    fprintf('Heatmap generated successfully.\n');
else
    fprintf('No neurons met the firing rate threshold of %.2f Hz.\n', firingRateThreshold);
end

%% Categorizing neurons based on their activity around stimulus
% Initialize matrices to store the firing rate data and original IDs for each category
alwaysFiring_neurons = []; alwaysFiring_ids = [];
alwaysSilent_neurons = []; alwaysSilent_ids = [];
preStim_neurons = []; preStim_ids = [];
postStim_neurons = []; postStim_ids = [];

% Loop through each neuron to categorize it
for neuronID = 1:nClusters
    % Use the full binned array for categorization, as it represents all trials
    fullSpikeMatrix = spikes(neuronID).binnedArray;
    
    % Categorize the neuron based on its overall activity
    category = categorizeNeuronActivity(fullSpikeMatrix);
    
    % Append the neuron's average firing rate data and its ID to the correct matrix
    switch category
        case 'Always Firing'
            alwaysFiring_neurons = [alwaysFiring_neurons; averageFiringRates(neuronID, :)];
            alwaysFiring_ids = [alwaysFiring_ids; clusterIDs(neuronID)];
        case 'Always Silent'
            alwaysSilent_neurons = [alwaysSilent_neurons; averageFiringRates(neuronID, :)];
            alwaysSilent_ids = [alwaysSilent_ids; clusterIDs(neuronID)];
        case 'Pre-Stimulus Firing'
            preStim_neurons = [preStim_neurons; averageFiringRates(neuronID, :)];
            preStim_ids = [preStim_ids; clusterIDs(neuronID)];
        case 'Post-Stimulus Firing'
            postStim_neurons = [postStim_neurons; averageFiringRates(neuronID, :)];
            postStim_ids = [postStim_ids; clusterIDs(neuronID)];
    end
end

% Display Categorization Results
fprintf('\n--- Neuron Activity Categorization Summary ---\n');
fprintf('Always Firing:         %d neurons\n', size(alwaysFiring_neurons, 1));
fprintf('Always Silent:         %d neurons\n', size(alwaysSilent_neurons, 1));
fprintf('Pre-Stimulus Firing:   %d neurons\n', size(preStim_neurons, 1));
fprintf('Post-Stimulus Firing:  %d neurons\n', size(postStim_neurons, 1));


%% Plot Heatmaps for each Neuron Category
figure('Name', 'Heatmaps by Neuron Category', 'Position', [100 100 1200 800]);
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, 'Neuronal Firing Rate Tuning by Category');

% Plot "Always Firing"
if ~isempty(alwaysFiring_neurons)
    nexttile; % Select the tile first
    h1 = heatmap(unique_trials_ids, string(alwaysFiring_ids), alwaysFiring_neurons);
    h1.Title = sprintf('Always Firing (%d neurons)', size(alwaysFiring_neurons,1));
    h1.Colormap = jet;
    h1.XLabel = 'Contrast'; h1.YLabel = 'Neuron ID';
end

% Plot "Post-Stimulus Firing"
if ~isempty(postStim_neurons)
    nexttile; % Select the tile first
    h2 = heatmap(unique_trials_ids, string(postStim_ids), postStim_neurons);
    h2.Title = sprintf('Post-Stimulus Firing (%d neurons)', size(postStim_neurons,1));
    h2.Colormap = jet;
    h2.XLabel = 'Contrast'; h2.YLabel = 'Neuron ID';
end

% Plot "Pre-Stimulus Firing"
if ~isempty(preStim_neurons)
    nexttile; % Select the tile first
    h3 = heatmap(unique_trials_ids, string(preStim_ids), preStim_neurons);
    h3.Title = sprintf('Pre-Stimulus Firing (%d neurons)', size(preStim_neurons,1));
    h3.Colormap = jet;
    h3.XLabel = 'Contrast'; h3.YLabel = 'Neuron ID';
end

% Plot "Always Silent"
if ~isempty(alwaysSilent_neurons)
    nexttile; % Select the tile first
    h4 = heatmap(unique_trials_ids, string(alwaysSilent_ids), alwaysSilent_neurons);
    h4.Title = sprintf('Always Silent (%d neurons)', size(alwaysSilent_neurons,1));
    h4.Colormap = jet;
    h4.XLabel = 'Contrast'; h4.YLabel = 'Neuron ID';
end



%% Trying another approach for tuning curves
%% Pre-calculate data for Superplot
fprintf('\nPre-calculating data for superplot...\n');
timeVector = (win(1) + binSize/2) : binSize : (win(2) - binSize/2);
numTimeBins = length(timeVector);
activity_matrix_all_trials = zeros(nClusters, trialTypes, numTimeBins);
peak_firing_rates = zeros(nClusters, trialTypes);

for neuronID = 1:nClusters
    neuron_spikes = spikes(neuronID).binnedArray;
    for trialidx = 1:trialTypes
        trialType = unique_trials_ids(trialidx);
        trial_only = find((trials.contrast == trialType) & (trials.isStimulus == 1));
        spikeMatrix_subset = neuron_spikes(trial_only,:);
        
        if ~isempty(spikeMatrix_subset)
            num_trials_in_subset = size(spikeMatrix_subset, 1);
            spikeCounts_per_bin = sum(spikeMatrix_subset, 1);
            psth_rate = mean(spikeCounts_per_bin,1); % Firing rate in Hz
            activity_matrix_all_trials(neuronID, trialidx, :) = psth_rate;
            peak_firing_rates(neuronID, trialidx) = max(psth_rate);
        else
            activity_matrix_all_trials(neuronID, trialidx, :) = zeros(1, numTimeBins);
        end
    end
end
fprintf('Pre-calculation for superplot finished.\n');


%% Plot Superplot of Sorted Activity by Trial Type
fprintf('\nGenerating superplot of sorted activity by trial type...\n');

figure('Name', 'Sorted Activity by Trial Type', 'Position', [100 100 1400 800]);
% Create a layout that fits 9 plots well, e.g., 3x3
superplot_layout = tiledlayout(1, trialTypes, 'TileSpacing', 'compact', 'Padding', 'normal');
title(superplot_layout, 'Neuronal Activity Sorted Independently for Each Trial Type');

% Get global color limits for consistent scaling across subplots
cmin = min(activity_matrix_all_trials, [], 'all');
cmax = max(activity_matrix_all_trials, [], 'all');

% Loop through each trial type to create one subplot per type
for trialidx = 1:trialTypes
    % Get the activity for all neurons for this specific trial type
    activity_this_trial = squeeze(activity_matrix_all_trials(:, trialidx, :));
    
    % Get the peak rates for this trial to sort by
    peaks_this_trial = peak_firing_rates(:, trialidx);
    
    % Sort neurons based on their peak firing rate *for this trial type*
    [~, sortedNeuronOrder] = sort(peaks_this_trial, 'descend');
    
    % Reorder the activity matrix based on the sort order
    sorted_activity = activity_this_trial(sortedNeuronOrder, :);
    
    % Select the next subplot
    ax = nexttile;
    
    % Use imagesc to create a heatmap of the sorted activity
    imagesc(ax, timeVector, 1:nClusters, sorted_activity);
    hold(ax, 'on');
    
    % Add a vertical line at t=0
    xline(ax, 0, '--w', 'LineWidth', 1.5); % White line for better visibility
    hold(ax, 'off');
    
    % Formatting
    title(ax, sprintf('Contrast: %.2f', unique_trials_ids(trialidx)));
    colormap(ax, 'jet'); 
    clim(ax, [cmin cmax]); % Apply consistent color limits
    set(ax, 'YDir', 'normal'); % Ensure neuron 1 is at the bottom
    xlabel(ax, 'Time (s)');
    
    % Set y-axis to show actual neuron IDs with a smaller font
    yticks(ax, 1:nClusters);
    yticklabels(ax, string(clusterIDs(sortedNeuronOrder)));
    ax.YAxis.FontSize = 2; % Reduce font size to fit labels
    ylabel(ax, 'Neuron ID (Sorted)');
end
%% Plot Superplot of Top 50 Sorted Neurons by Trial Type
fprintf('\nGenerating superplot of top 50 sorted neurons by trial type...\n');

figure('Name', 'Top 50 Sorted Neurons by Trial Type', 'Position', [100 100 1600 300]);
% Create a layout that fits 9 plots well, e.g., 3x3
superplot_layout = tiledlayout(1, 9, 'TileSpacing', 'compact', 'Padding', 'normal');
title(superplot_layout, 'Tuning Curve '+ regions.name(areaID)+' Session 2');

% Get global color limits for consistent scaling across subplots
cmin = min(activity_matrix_all_trials, [], 'all');
cmax = max(activity_matrix_all_trials, [], 'all');

n_top_neurons = 20; % Define the number of top neurons to plot

% Loop through each trial type to create one subplot per type
for trialidx = 1:trialTypes
    % Get the activity for all neurons for this specific trial type
    activity_this_trial = squeeze(activity_matrix_all_trials(:, trialidx, :));
    
    % Get the peak rates for this trial to sort by
    peaks_this_trial = peak_firing_rates(:, trialidx);
    
    % Sort neurons based on their peak firing rate *for this trial type*
    [~, sortedNeuronOrder] = sort(peaks_this_trial, 'descend');
    
    % --- Select only the top N neurons ---
    num_to_plot = min(n_top_neurons, nClusters); % Ensure we don't exceed the number of available neurons
    top_neuron_indices = sortedNeuronOrder(1:num_to_plot);
    
    % Reorder the activity matrix based on the top N sort order
    sorted_activity = activity_this_trial(top_neuron_indices, :);
    
    % Select the next subplot
    ax = nexttile;
    
    % Use imagesc to create a heatmap of the sorted activity
    imagesc(ax, timeVector, 1:num_to_plot, sorted_activity);
    hold(ax, 'on');
    
    % Add a vertical line at t=0
    xline(ax, 0, '--w', 'LineWidth', 1.5); % White line for better visibility
    hold(ax, 'off');
    
    % Formatting
    title(ax, sprintf('Contrast: %.2f', unique_trials_ids(trialidx)));
    colormap(ax, 'jet'); 
    clim(ax, [cmin cmax]); % Apply consistent color limits
    set(ax, 'YDir', 'normal'); % Ensure neuron 1 is at the bottom
    xlabel(ax, 'Time (s)');
    
    % Set y-axis to show actual neuron IDs with a smaller font
    yticks(ax, 1:num_to_plot);
    yticklabels(ax, string(clusterIDs(top_neuron_indices)));
    ax.YAxis.FontSize = 0.5; % Reduce font size to fit labels
    ylabel(ax, 'Neuron ID (Sorted)');
end

% Add a single colorbar for the entire layout
cb = colorbar;
cb.Layout.Tile = 'east'; % Place it on the east side of the layout
ylabel(cb, 'Average number of spikes per trial');

saveas(gcf,'Tuning Curve ' + regions.name(areaID) +' Session 2', 'png')
fprintf('Superplot of top 50 sorted neurons generated successfully.\n');


%%

% =========================================================================
% Functions
% =========================================================================

function rasterplot_naman(ax, spikeMatrix, win, binSize)
% rasterplot_naman Generates a raster plot on a specified axes object.
%
%   Args:
%       ax (axes handle): The axes object where the plot will be drawn.
%       spikeMatrix (matrix): A 2D matrix where rows are trials and
%                             columns are time bins.
%       win (vector): A 2-element vector [startTime, endTime] for the plot window.
%       binSize (scalar): The size of each time bin.

    if isempty(spikeMatrix)
        return; % Do nothing if there are no trials
    end

    % --- Time Vector Definition ---
    [numTrials, numTimeBins] = size(spikeMatrix);
    startTime = win(1);
    timeVector = startTime + (0:numTimeBins-1) * binSize;

    % --- Plotting Setup ---
    hold(ax, 'on');

    % --- Generate Raster Plot ---
    for trial = 1:numTrials
        spikeTimesInTrial = timeVector(spikeMatrix(trial, :) > 0);
        if ~isempty(spikeTimesInTrial)
            % Plot all spikes for a given trial at once for efficiency
            plot(ax, [spikeTimesInTrial; spikeTimesInTrial], [trial-0.4; trial+0.4], 'k', 'LineWidth', 1.0);
        end
    end

    % --- Aesthetics and Labels ---
    xline(ax, 0, '--', 'r', 'LineWidth', 1);
    hold(ax, 'off');
    xlim(ax, win);
    ylim(ax, [0, numTrials + 1]);
end
