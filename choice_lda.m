% =========================================================================
% Main Script: Time-Resolved Decoding of Choice from Neural Activity
% =========================================================================
close all;
clear all;
clc;
% --- Step 1: Load Data and Define Parameters ---
sesPath = '../data/Steinmetz/Hench_2017-06-17';
[S, ~, neurons, trials] = stOpenSession(sesPath);
%%
areaID = 11; % e.g., VISp
win = [-0.2, 0.5]; % Define the full analysis window relative to stimulus onset
binSize = 0.02;

% --- Step 2: Prepare the Data ---
% We need two main things:
% 1. A response vector (Y): The animal's choices.
% 2. A feature matrix (X): The neural activity for each trial.

% Get the animal's choices. The 'response_choice' field is typically:
% -1 for a left choice, 1 for a right choice, and 0 for no choice (no-go).
% We must filter out the 'no-go' trials for a binary classification.
valid_trials_idx = S.trials.response_choice ~= 0;
Y_choice = S.trials.response_choice(valid_trials_idx);

% Get the neural activity for the valid trials.
clusterIDs = find(neurons.region == areaID);
nClusters = length(clusterIDs);
nValidTrials = sum(valid_trials_idx);
timeVector = win(1):binSize:win(2);
nTimeBins = length(timeVector) - 1;

%% Create a 3D matrix using a vectorized approach
fprintf('Creating 3D activity matrix with vectorized loop...\n');

% First, get only the stimulus times for the trials we need
valid_stim_times = trials.visStimTime(valid_trials_idx);

% Pre-allocate the final 3D matrix for efficiency
X_activity = zeros(nValidTrials, nClusters, nTimeBins);

% Loop only through the neurons (the trial loop is now vectorized)
for neuronID = 1:nClusters
    % Get all spike times for the current neuron
    current_neuron_spikes = S.spikes.times(S.spikes.clusters == clusterIDs(neuronID));
    
    % Call psthAndBA ONCE for this neuron, passing all valid trial times at once.
    % The output 'binnedArray' will be a 2D matrix of [nValidTrials x nTimeBins].
    [~, ~, ~, ~, ~, binnedArray] = psthAndBA(current_neuron_spikes, valid_stim_times, win, binSize);
    
    % Assign this entire 2D slice of data to the 3D matrix
    X_activity(:, neuronID, :) = binnedArray;
end
%%
% --- NEW: Use PCA on Trial-Averaged Data to find Robust Components ---
fprintf('Performing PCA on trial-averaged population activity...\n');
n_PCs = 10; % Define how many principal components to use

% 1. Create the trial-averaged matrix: [Neurons x Time Bins]
X_mean_across_trials = squeeze(mean(X_activity, 1));

% 2. Perform PCA on the transpose of this matrix.
%    The input is [Time Bins x Neurons]. The components ('coeff') will be the
%    principal modes of population co-activation, or "eigen-neurons".
[coeff, ~, ~, ~, explained] = pca(X_mean_across_trials', 'NumComponents', n_PCs);

fprintf('PCA on trial-averaged data complete. Top %d components explain %.2f%% of the variance.\n', n_PCs, sum(explained(1:n_PCs)));

% 3. Project the original, noisy single-trial data onto these robust components.
%    First, reshape the full activity matrix so neurons are in the columns.
X_reshaped = reshape(X_activity, nValidTrials * nTimeBins, nClusters);
%    Then, project onto the components to get the scores.
score = X_reshaped * coeff;

% 4. Reshape the scores back into a 3D matrix: Trials x PCs x Time Bins
X_pca_activity = reshape(score, nValidTrials, nTimeBins, n_PCs);
X_pca_activity = permute(X_pca_activity, [1 3 2]); % Reorder to Trials x PCs x Time Bins

%%
fprintf('Activity matrix created successfully.\n');
% --- NEW: Create the Mean Population Activity Matrix ---
% Instead of using every neuron, we now average across them.
% The result is a 2D matrix: Trials x Time Bins
fprintf('Calculating mean population activity...\n');
X_mean_activity = squeeze(mean(X_activity, 2));

%% --- Step 3: Incremental Window Decoding using Mean Activity ---
fprintf('Starting incremental window decoding in parallel...\n');

% We will store the accuracy at each time point
lda_accuracy_over_time = zeros(1, nTimeBins);
logreg_accuracy_over_time = zeros(1, nTimeBins);

% Convert Y_choice to a 0/1 variable for the logistic regression model
Y_choice_01 = (Y_choice + 1) / 2;

% --- OPTIMIZATION: Create a constant object for the broadcast variable ---
% This sends the data to the workers only once to reduce communication overhead.
%X_mean_const = parallel.pool.Constant(X_mean_activity);
X_const=parallel.pool.Constant(X_pca_activity);
% Use a parallel for-loop (parfor) to run iterations simultaneously
parfor t = 1:nTimeBins
    % The feature matrix is now simply the first 't' time bins of the mean activity.
    % We access the data from the constant object using .Value
    %X_features = X_mean_const.Value(:, 1:t);
    X_features = X_const.Value(:, 1:t);
    % --- Train and Test LDA Model ---
    % 'diagLinear' handles potential zero-variance issues in early time bins
    lda_cv_model = fitcdiscr(X_features, Y_choice, 'CrossVal', 'on', 'KFold', 5, 'DiscrimType', 'diagLinear');
    lda_accuracy_over_time(t) = 1 - kfoldLoss(lda_cv_model);
    
    % --- Train and Test Logistic Regression Model using fitclinear ---
    % fitclinear is fast and robust for this type of data.
    logreg_cv_model = fitclinear(X_features, Y_choice_01, 'Learner', 'logistic', 'CrossVal', 'on', 'KFold', 5);
    logreg_accuracy_over_time(t) = 1 - kfoldLoss(logreg_cv_model);
    
end

fprintf('Decoding finished.\n');

%% --- Step 4: Plot the Results ---
fprintf('Plotting decoding accuracy over time...\n');
figure('Name', 'Time-Resolved Decoding of Choice from Population Activity');
hold on;
plot(timeVector(1:nTimeBins), lda_accuracy_over_time, 'LineWidth', 2, 'DisplayName', 'LDA');
plot(timeVector(1:nTimeBins), logreg_accuracy_over_time, 'LineWidth', 2, 'DisplayName', 'Logistic Regression');
hold off;
grid on;
xlabel('Time from Stimulus Onset (s)');
ylabel('Decoding Accuracy');
title('Predicting Animal Choice from Mean Population Activity');
legend('show', 'Location', 'southeast');
ylim([0.4 1.0]); % Set reasonable y-axis limits, chance is 0.5
% Add a line for chance level
yline(0.5, '--k', 'Chance');
%%
% --- Verification Step: Plot mean activity for each choice ---
figure('Name', 'Mean Activity by Choice');
hold on;
left_trials = Y_choice == -1;
right_trials = Y_choice == 1;

plot(timeVector(1:nTimeBins), mean(X_mean_activity(left_trials, :), 1), 'b', 'LineWidth', 2, 'DisplayName', 'Left Choice');
plot(timeVector(1:nTimeBins), mean(X_mean_activity(right_trials, :), 1), 'r', 'LineWidth', 2, 'DisplayName', 'Right Choice');
hold off;
grid on;
title('Average Population Activity by Animal''s Choice');
xlabel('Time from Stimulus Onset (s)');
ylabel('Mean Firing Rate (Hz)');
legend('show');