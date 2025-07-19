
function category = categorizeNeuronActivity(spikeMatrix)
% categorizeNeuronActivity Classifies a neuron's firing pattern relative
% to a stimulus at t=0.

if isempty(spikeMatrix)
    category = 'Always Silent';
    return;
end

% --- Define Time Parameters and Windows ---
[numTrials, numTimeBins] = size(spikeMatrix);
startTime = -2;
binSize = 0.02;
timeVector = startTime + (0:numTimeBins-1) * binSize;

% Define pre-stimulus and post-stimulus windows
preStimIndices = timeVector < 0;
postStimIndices = timeVector >= 0;

preStimDuration = 2.0; % -2s to 0s
postStimDuration = 2.5; % 0s to 2.5s

% --- Calculate Firing Rates for Each Window ---
if numTrials > 0
    preStimSpikes = sum(spikeMatrix(:, preStimIndices), 'all');
    postStimSpikes = sum(spikeMatrix(:, postStimIndices), 'all');

    preStimRate = preStimSpikes / (numTrials * preStimDuration);
    postStimRate = postStimSpikes / (numTrials * postStimDuration);
else
    preStimRate = 0;
    postStimRate = 0;
end

% --- Categorize Based on Firing Rates ---
activityThreshold = 3; % Firing rate in Hz to be considered 'active'

is_active_before = preStimRate >= activityThreshold;
is_active_after = postStimRate >= activityThreshold;

if is_active_before && is_active_after
    category = 'Always Firing';
elseif ~is_active_before && ~is_active_after
    category = 'Always Silent';
elseif is_active_before && ~is_active_after
    category = 'Pre-Stimulus Firing';
else % ~is_active_before && is_active_after
    category = 'Post-Stimulus Firing';
end
