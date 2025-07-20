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

%%
spikes_temp = struct();
for idx = 1:nClusters
    spikes_temp(idx).spiketimes = S.spikes.times(S.spikes.clusters(:) == clusterIDs(idx));
end

% First, aggregate all spike times from the selected neurons into one vector for efficiency
all_cluster_spiketimes = cat(1, spikes_temp.spiketimes);
all_cluster_spiketimes = sort(all_cluster_spiketimes);

% Initialize a structure to store the results for each trial
trial_activity = struct();

% Define a relative time vector for the analysis window (e.g., -0.5s to +0.5s)
% This vector represents the center time of each bin relative to stimulus onset
relative_time_vector = (win(1) + binSize/2) : binSize : (win(2) - binSize/2);

% Loop through each trial
for k = 1:trials.N
    % Define the time window for the current trial based on its stimulus time
    time_window_start = stimTimes(k) + win(1);
    time_window_end = stimTimes(k) + win(2);
    
    % Define the precise bin edges for this trial's window
    bin_edges = time_window_start:binSize:time_window_end;
    
    % Find all spikes from the cluster that fall within this trial's window
    spikes_in_window = all_cluster_spiketimes(all_cluster_spiketimes >= time_window_start & all_cluster_spiketimes < time_window_end);
    
    % Count the total number of spikes in each time bin
    spike_counts_per_bin = histcounts(spikes_in_window, bin_edges);
    
    % Calculate the mean firing rate in Hz
    % Formula: (Total Spikes in Bin / Number of Neurons) / Bin Duration
    mean_firing_rate_hz = spike_counts_per_bin / nClusters / binSize;
    
    % Save the results for the current trial
    trial_activity(k).stimTime = stimTimes(k);
    trial_activity(k).respTime = respTimes(k);
    trial_activity(k).goCueTime = goTimes(k);
    trial_activity(k).timeVector = relative_time_vector;
    trial_activity(k).meanFiringRate = mean_firing_rate_hz;
end

fprintf('Calculation complete. The "trial_activity" structure now holds the results.\n');

%% Verification: Display results for the first trial and plot a few examples
disp('--- Example data from the first trial ---');
disp(trial_activity(1));

figure('Name', 'Mean Cluster Activity for First 5 Trials');
hold on;
for i = 1:5
    plot(trial_activity(i).timeVector, trial_activity(i).meanFiringRate, 'LineWidth', 1.5);
end
hold off;
grid on;
title('Mean Cluster Activity for First 5 Trials');
xlabel('Time from Stimulus Onset (s)');
ylabel('Mean Firing Rate (Hz)');
legend('Trial 1', 'Trial 2', 'Trial 3', 'Trial 4', 'Trial 5');
%% Convert the trial activity structure to a matrix
fprintf('Converting structure to matrix...\n');

% Determine the dimensions of the matrix
num_trials = trials.N;
num_time_bins = length(trial_activity(1).timeVector);

% Pre-allocate the matrix for efficiency
activity_matrix = zeros(num_trials, num_time_bins);

% Loop through each trial and populate the matrix
for k = 1:num_trials
    % Ensure the row vector from the structure fits the matrix dimensions
    if length(trial_activity(k).meanFiringRate) == num_time_bins
        activity_matrix(k, :) = trial_activity(k).meanFiringRate;
    end
end

fprintf('Matrix created successfully.\n');

%% Verification: Display the size of the new matrix and plot it
fprintf('The dimensions of the activity matrix are: %d rows (trials) x %d columns (time bins)\n', size(activity_matrix, 1), size(activity_matrix, 2));

figure('Name', 'Heatmap of Mean Cluster Activity Across All Trials');
imagesc(trial_activity(1).timeVector, 1:num_trials, activity_matrix);
colorbar;
xlabel('Time from Stimulus Onset (s)');
ylabel('Trial Number');
title('Mean Cluster Activity Across All Trials');