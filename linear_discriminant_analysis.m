
% =========================================================================
% Main Script: LDA for Neuron Classification by Session and Area
% =========================================================================

% --- Step 1: Define Datasets and Parameters ---
% Define the paths and area IDs for the sessions and areas you want to compare.
close all;
clear all;
clc

session1_path = '../data/Steinmetz/Hench_2017-06-17';
session1_area1_ID = 11; % e.g., VISp
session1_area2_ID = 1; % e.g., CA1

session2_path = '../data/Steinmetz/Cori_2016-12-18';
session2_area1_ID = 10; % e.g., VISp
session2_area2_ID = 1; % e.g., CA1

win = [-0.5, 0.5]; % Analysis window
binSize = 0.02;    % Bin size

%% --- Step 2: Load Data for Both Sessions ---
fprintf('Loading data for Session 1...\n');
[S1, ~, neurons1, trials1] = stOpenSession(session1_path);

fprintf('Loading data for Session 2...\n');
[S2, ~, neurons2, trials2] = stOpenSession(session2_path);

%% --- Step 3: Process Each Session/Area Combination ---
% We will call a helper function for each of the four groups, passing the loaded data.

fprintf('Processing Session 1, Area 1...\n');
[activity1, labels1] = process_session_area(S1, neurons1, trials1, session1_area1_ID, win, binSize, 'S1-A1');

fprintf('Processing Session 1, Area 2...\n');
[activity2, labels2] = process_session_area(S1, neurons1, trials1, session1_area2_ID, win, binSize, 'S1-A2');

fprintf('Processing Session 2, Area 1...\n');
[activity3, labels3] = process_session_area(S2, neurons2, trials2, session2_area1_ID, win, binSize, 'S2-A1');

fprintf('Processing Session 2, Area 2...\n');
[activity4, labels4] = process_session_area(S2, neurons2, trials2, session2_area2_ID, win, binSize, 'S2-A2');

%% --- Step 4: Combine Data from All Four Groups ---
% Vertically stack the activity matrices and corresponding labels.
combined_activity_matrix = [activity1; activity2; activity3; activity4];
combined_group_labels = [labels1; labels2; labels3; labels4];

fprintf('Combined data from %d total neurons across 4 groups.\n', length(combined_group_labels));

%% --- Step 4.5: Data Validation Plots ---
fprintf('Generating data validation plots...\n');

% Plot 1: Heatmap of all neuron activities
figure('Name', 'All Neuron Activity Before LDA');
timeVector = (win(1) + binSize/2) : binSize : (win(2) - binSize/2);
imagesc(timeVector, 1:size(combined_activity_matrix,1), combined_activity_matrix);
xlabel('Time from Stimulus Onset (s)');
ylabel('Neuron Index (All Groups)');
title('Heatmap of Average Firing Rate for All Neurons');
colorbar;

% Plot 2: Mean activity trace for each group
figure('Name', 'Mean Activity by Group');
hold on;
group_names = unique(combined_group_labels);
colors = lines(length(group_names)); % Get distinct colors for each group
for i = 1:length(group_names)
    current_group = group_names{i};
    group_indices = strcmp(combined_group_labels, current_group);
    mean_activity = mean(combined_activity_matrix(group_indices, :), 1);
    plot(timeVector, mean_activity, 'LineWidth', 2, 'Color', colors(i,:), 'DisplayName', current_group);
end
hold off;
grid on;
title('Mean Firing Activity for Each Group');
xlabel('Time from Stimulus Onset (s)');
ylabel('Average Firing Rate (Hz)');
legend('show', 'Location', 'best');


%% --- Step 5: Perform Linear Discriminant Analysis ---
% Project the data onto the first three discriminant axes to see the separation.
[~, scores] = predict(LDA_model, combined_activity_matrix);

fprintf('Plotting LDA results in 3D...\n');
figure('Name', '3D LDA of Neuron Origin', 'Position', [100 100 1000 800]);
hold on;
group_names_for_plot = unique(combined_group_labels);
colors_for_plot = lines(length(group_names_for_plot));
for i = 1:length(group_names_for_plot)
    current_group = group_names_for_plot{i};
    group_indices = strcmp(combined_group_labels, current_group);
    % Use scatter3 for 3D plotting
    scatter3(scores(group_indices,1), scores(group_indices,2), scores(group_indices,3), 36, colors_for_plot(i,:), 'filled', 'DisplayName', current_group);
end
hold off;
view(3); % Set the default 3D view
title('3D Linear Discriminant Analysis of Neuron Origin (Session & Area)');
xlabel('Linear Discriminant 1');
ylabel('Linear Discriminant 2');
zlabel('Linear Discriminant 3');
legend('Location', 'best');
grid on;
axis equal;

%% --- Step 7 (Optional): Quantify Classification Accuracy ---
% Use a confusion matrix to see how well the model can distinguish the groups.
fprintf('Calculating confusion matrix...\n');
predicted_labels = predict(LDA_model, combined_activity_matrix);

figure('Name', 'LDA Classification Accuracy by Origin');
confusionchart(combined_group_labels, predicted_labels);
title('LDA Model Confusion Matrix (by Origin)');

% =========================================================================
% Helper Function: process_session_area
% =========================================================================
% This function takes loaded data for one session, extracts the activity for one
% specified brain area, and assigns the correct label to each neuron.

function [neuron_activity_matrix, group_labels] = process_session_area(S, neurons, trials, areaID, win, binSize, label_string)
    
    % --- Find neurons for the specified area ---
    clusterIDs = find(neurons.region == areaID);
    nClusters = length(clusterIDs);
    if isempty(clusterIDs)
        neuron_activity_matrix = [];
        group_labels = {};
        fprintf('Warning: No neurons found for area ID %d.\n', areaID);
        return;
    end
    
    % --- Create 'spikes' structure for the specified area ---
    spikes = struct();
    for idx = 1:nClusters
        spikes(idx).clu = clusterIDs(idx);
        spikes(idx).spiketimes = S.spikes.times(S.spikes.clusters(:) == spikes(idx).clu);
        % CORRECTED LINE: Use the 'trials' structure passed into the function
        [~, ~, ~, ~, ~, spikes(idx).binnedArray] = psthAndBA(spikes(idx).spiketimes, trials.visStimTime, win, binSize);
    end
    
    % --- Calculate Neuron Activity Matrix (each row is a neuron) ---
    num_time_bins = size(spikes(1).binnedArray, 2);
    neuron_activity_matrix = zeros(nClusters, num_time_bins);
    for neuronID = 1:nClusters
        if ~isempty(spikes(neuronID).binnedArray)
            neuron_activity_matrix(neuronID, :) = mean(spikes(neuronID).binnedArray, 1);
        end
    end
    
    % --- Create Group Labels ---
    % Assign the same descriptive label to every neuron from this group.
    group_labels = repmat({label_string}, nClusters, 1);
end
