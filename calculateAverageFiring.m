function avgFiringRate = calculateAverageFiring(spikeMatrix)
% calculateAverageFiring Calculates the mean firing rate across trials
% in the time window from 0s to 1s.
%
%   Args:
%       spikeMatrix (matrix): A 2D matrix where rows are trials and
%                             columns are time bins.
%
%   Returns:
%       avgFiringRate (double): The calculated average firing rate in Hz (spikes/sec).

% --- Define Time Parameters ---
[~, numTimeBins] = size(spikeMatrix);
startTime = -2;
binSize = 0.02;
endTime = 2.5;
timeVector = startTime + (0:numTimeBins-1) * binSize;

% --- Identify Bins of Interest (0s to 1s) ---
% Find the logical indices for the time bins between 0 and 1 second.
timeWindowIndices = timeVector >= 0 & timeVector <= endTime;

% --- Calculate Firing Rate ---
% Extract the part of the matrix corresponding to our time window.
spikesInWindow = spikeMatrix(:, timeWindowIndices);

% Sum all spikes across all trials within this window.
totalSpikes = sum(spikesInWindow, 'all');

% Get the number of trials.
numTrials = size(spikeMatrix, 1);

% Define the duration of the time window in seconds.
windowDuration = endTime; % 1s - 0s = 1s

% Calculate the average firing rate in spikes/second (Hz).
% Formula: (Total Spikes) / (Number of Trials * Duration of Window)
avgFiringRate = totalSpikes / (numTrials * windowDuration);

end
