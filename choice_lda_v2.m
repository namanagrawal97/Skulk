% =========================================================================
% Main Script: Time-Resolved Decoding of Choice from PCA of Population Activity
% =========================================================================

% --- Step 1: Load Data and Define Parameters ---
clear all; close all; clc
sesPath = '../data/Steinmetz/Hench_2017-06-17';
[S, ~, neurons, trials] = stOpenSession(sesPath);

areaID = 11; % e.g., VISp
win = [-0.2, 0.5]; % Define the full analysis window relative to stimulus onset
binSize = 0.02;

% =========================================================================
% Analysis 1: Predict Left vs. Right Choice
% =========================================================================

% --- Step 2a: Prepare Data for Left vs. Right Choice ---
fprintf('--- Starting Analysis 1: Left vs. Right Choice Prediction ---\n');
% We must filter out the 'no-go' trials for a binary classification.
valid_trials_idx = S.trials.response_choice ~= 0;
Y_choice = S.trials.response_choice(valid_trials_idx);

% Get the neural activity for the valid trials.
clusterIDs = find(neurons.region == areaID);
nClusters = length(clusterIDs);
nValidTrials = sum(valid_trials_idx);
timeVector = win(1):binSize:win(2);
nTimeBins = length(timeVector) - 1;

% Create a 3D matrix: Trials x Neurons x Time Bins (Vectorized)
fprintf('Creating 3D activity matrix for choice trials...\n');
valid_stim_times = trials.visStimTime(valid_trials_idx);
X_activity_choice = zeros(nValidTrials, nClusters, nTimeBins);
for neuronID = 1:nClusters
    current_neuron_spikes = S.spikes.times(S.spikes.clusters == clusterIDs(neuronID));
    [~, ~, ~, ~, ~, binnedArray] = psthAndBA(current_neuron_spikes, valid_stim_times, win, binSize);
    X_activity_choice(:, neuronID, :) = binnedArray;
end
fprintf('Activity matrix created successfully.\n');

% --- Use PCA on Trial-Averaged Data to find Robust Components ---
fprintf('Performing PCA on trial-averaged choice data...\n');
n_PCs = 10; % Define how many principal components to use
X_mean_across_trials = squeeze(mean(X_activity_choice, 1));
[coeff, ~, ~, ~, explained] = pca(X_mean_across_trials', 'NumComponents', n_PCs);
fprintf('PCA on trial-averaged data complete. Top %d components explain %.2f%% of the variance.\n', n_PCs, sum(explained(1:n_PCs)));
X_reshaped = reshape(X_activity_choice, nValidTrials * nTimeBins, nClusters);
score = X_reshaped * coeff;
X_pca_activity_choice = reshape(score, nValidTrials, nTimeBins, n_PCs);
X_pca_activity_choice = permute(X_pca_activity_choice, [1 3 2]); % Reorder to Trials x PCs x Time Bins

% --- Step 3a: Incremental Window Decoding for Left vs. Right Choice ---
fprintf('Starting incremental window decoding for choice...\n');
lda_accuracy_over_time = zeros(1, nTimeBins);
logreg_accuracy_over_time = zeros(1, nTimeBins);
Y_choice_01 = (Y_choice + 1) / 2;
X_pca_const = parallel.pool.Constant(X_pca_activity_choice);

parfor t = 1:nTimeBins
    current_window_data = X_pca_const.Value(:, :, 1:t);
    n_features = n_PCs * t;
    X_features = reshape(current_window_data, nValidTrials, n_features);
    lda_cv_model = fitcdiscr(X_features, Y_choice, 'CrossVal', 'on', 'KFold', 5, 'DiscrimType', 'diagLinear');
    lda_accuracy_over_time(t) = 1 - kfoldLoss(lda_cv_model);
    logreg_cv_model = fitclinear(X_features, Y_choice_01, 'Learner', 'logistic', 'CrossVal', 'on', 'KFold', 5);
    logreg_accuracy_over_time(t) = 1 - kfoldLoss(logreg_cv_model);
end
fprintf('Choice decoding finished.\n');

% --- Step 4a: Plot the Results for Left vs. Right Choice ---
fprintf('Plotting decoding accuracy over time for choice...\n');
figure('Name', 'Time-Resolved Decoding of Left vs. Right Choice');
hold on;
plot(timeVector(1:nTimeBins), lda_accuracy_over_time, 'LineWidth', 2, 'DisplayName', 'LDA');
plot(timeVector(1:nTimeBins), logreg_accuracy_over_time, 'LineWidth', 2, 'DisplayName', 'Logistic Regression');
hold off;
grid on;
xlabel('Time from Stimulus Onset (s)');
ylabel('Decoding Accuracy');
title(sprintf('Predicting Animal Choice (Left vs. Right) from Top %d PCs', n_PCs));
legend('show', 'Location', 'southeast');
ylim([0.4 1.0]);
yline(0.5, '--k', 'Chance');


% =========================================================================
% Analysis 2: Predict Go vs. No-Go Trials
% =========================================================================
fprintf('\n--- Starting Analysis 2: Go vs. No-Go Prediction ---\n');

% --- Step 2b: Prepare Data for Go vs. No-Go ---
% The response vector is now 1 for a "go" trial and 0 for a "no-go" trial.
Y_gongo = (S.trials.response_choice ~= 0);
nAllTrials = trials.N;

% Create a 3D matrix for ALL trials (go and no-go)
fprintf('Creating 3D activity matrix for all trials...\n');
all_stim_times = trials.visStimTime;
X_activity_all = zeros(nAllTrials, nClusters, nTimeBins);
for neuronID = 1:nClusters
    current_neuron_spikes = S.spikes.times(S.spikes.clusters == clusterIDs(neuronID));
    [~, ~, ~, ~, ~, binnedArray] = psthAndBA(current_neuron_spikes, all_stim_times, win, binSize);
    X_activity_all(:, neuronID, :) = binnedArray;
end
fprintf('Activity matrix for all trials created successfully.\n');

% --- Use PCA on Trial-Averaged Data from ALL trials ---
fprintf('Performing PCA on trial-averaged data from all trials...\n');
X_mean_all_trials = squeeze(mean(X_activity_all, 1));
[coeff_gongo, ~, ~, ~, explained_gongo] = pca(X_mean_all_trials', 'NumComponents', n_PCs);
fprintf('PCA on all trials complete. Top %d components explain %.2f%% of the variance.\n', n_PCs, sum(explained_gongo(1:n_PCs)));
X_reshaped_all = reshape(X_activity_all, nAllTrials * nTimeBins, nClusters);
score_all = X_reshaped_all * coeff_gongo;
X_pca_activity_gongo = reshape(score_all, nAllTrials, nTimeBins, n_PCs);
X_pca_activity_gongo = permute(X_pca_activity_gongo, [1 3 2]); % Reorder to Trials x PCs x Time Bins

% --- Step 3b: Incremental Window Decoding for Go vs. No-Go ---
fprintf('Starting incremental window decoding for Go vs. No-Go...\n');
lda_gongo_accuracy = zeros(1, nTimeBins);
logreg_gongo_accuracy = zeros(1, nTimeBins);
X_pca_gongo_const = parallel.pool.Constant(X_pca_activity_gongo);

parfor t = 1:nTimeBins
    current_window_data = X_pca_gongo_const.Value(:, :, 1:t);
    n_features = n_PCs * t;
    X_features = reshape(current_window_data, nAllTrials, n_features);
    
    % LDA for Go/No-Go
    lda_cv_model = fitcdiscr(X_features, Y_gongo, 'CrossVal', 'on', 'KFold', 5, 'DiscrimType', 'diagLinear');
    lda_gongo_accuracy(t) = 1 - kfoldLoss(lda_cv_model);
    
    % Logistic Regression for Go/No-Go
    logreg_cv_model = fitclinear(X_features, Y_gongo, 'Learner', 'logistic', 'CrossVal', 'on', 'KFold', 5);
    logreg_gongo_accuracy(t) = 1 - kfoldLoss(logreg_cv_model);
end
fprintf('Go vs. No-Go decoding finished.\n');

% --- Step 4b: Plot the Results for Go vs. No-Go ---
fprintf('Plotting decoding accuracy over time for Go vs. No-Go...\n');
figure('Name', 'Time-Resolved Decoding of Go vs. No-Go');
hold on;
plot(timeVector(1:nTimeBins), lda_gongo_accuracy, 'LineWidth', 2, 'DisplayName', 'LDA');
plot(timeVector(1:nTimeBins), logreg_gongo_accuracy, 'LineWidth', 2, 'DisplayName', 'Logistic Regression');
hold off;
grid on;
xlabel('Time from Stimulus Onset (s)');
ylabel('Decoding Accuracy');
title(sprintf('Predicting Animal Engagement (Go vs. No-Go) from Top %d PCs', n_PCs));
legend('show', 'Location', 'southeast');
ylim([0.4 1.0]);
% Calculate and plot the correct chance level for imbalanced classes
chance_gongo = max(mean(Y_gongo), 1 - mean(Y_gongo));
yline(chance_gongo, '--k', sprintf('Chance (%.2f)', chance_gongo));

