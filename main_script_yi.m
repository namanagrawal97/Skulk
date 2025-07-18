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

%% Parameters to convert between frame numbers for behavior and time in seconds
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


%% PSTH (align to visual stimulus)
areaID = 10; % for area VISp
win = [-0.5,0.5]; % Original window, -0.5s to +0.5s around stimTime
binSize = 0.02;
selectedEvents = respTimes;
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


timeBinCenters = bins(1:end-1) + diff(bins)/2;
startPostStimBinIdx = find(timeBinCenters >= 0, 1, 'first');
if isempty(startPostStimBinIdx)
    warning('Could not find a bin center at or after 0. Check your win/binSize and bins calculation.');
    startPostStimBinIdx = 1; % Fallback to start from beginning if issue
end
endPostStimBinIdx = length(timeBinCenters); % End of the time series

% This is the range of time bins for your post-stimulus activity
postStimBinIndices = startPostStimBinIdx : endPostStimBinIdx;
numPostStimBins = length(postStimBinIndices); % This will dynamically be 25 in your case

%% Prepare data for PCA
% Assuming `spikes(idx).psth` has dimensions `numTrials x 50`
numTrials = size(spikes(1).psth, 1);
numTotalTimeBins = size(spikes(1).psth, 2); % This will be 50
% nClusters is already defined above

% Initialize the matrix for PCA with the correct number of columns
neuralDataForPCA = zeros(numTrials * nClusters, numPostStimBins);

for idx = 1:nClusters
    % Each block of rows corresponds to a single neuron's PSTH across all trials
    startRow = (idx - 1) * numTrials + 1;
    endRow = idx * numTrials;
    % Select only the post-stimulus part of the PSTH using the calculated indices
    neuralDataForPCA(startRow:endRow, :) = spikes(idx).psth(:, postStimBinIndices);
end

[coeff, score, latent, tsquared, explained, mu] = pca(neuralDataForPCA);

disp('PCA Results:');
disp(['Total variance explained by first 5 PCs: ', num2str(sum(explained(1:5))), '%']);
disp('Percentage of variance explained by each principal component:');
disp(explained'); % Display as a row vector for better readability


%% Visualization 1: Explained Variance (Scree Plot)
figure;
plot(1:length(explained), explained, 'o-');
xlabel('Principal Component Number');
ylabel('Percentage of Variance Explained');
title('Scree Plot: Explained Variance by Principal Components');
grid on;
hold on;
% Add a line for cumulative explained variance (optional but very useful)
cumulativeExplained = cumsum(explained);
plot(1:length(cumulativeExplained), cumulativeExplained, 'rx-', 'DisplayName', 'Cumulative Explained Variance');
legend('Individual PC Variance', 'Cumulative PC Variance');
hold off;

%% Visualization 2: The "Meta Neurons" (Principal Component Loadings/Coefficients)
% Make sure to use the *correct subset* of timeBinCenters corresponding to
% the data you fed into PCA.
timeBinsForPlotting = timeBinCenters(postStimBinIndices); % Only plot the post-stimulus part

figure;
subplot(3,1,1); % Plotting the first 3 principal components
plot(timeBinsForPlotting, coeff(:,1));
title('Principal Component 1 (PC1)');
xlabel('Time (s)');
ylabel('Loading');
grid on;

subplot(3,1,2);
plot(timeBinsForPlotting, coeff(:,2));
title('Principal Component 2 (PC2)');
xlabel('Time (s)');
ylabel('Loading');
grid on;

subplot(3,1,3);
plot(timeBinsForPlotting, coeff(:,3));
title('Principal Component 3 (PC3)');
xlabel('Time (s)');
ylabel('Loading');
grid on;

sgtitle('Principal Component Loadings (Meta Neuron Activity Patterns - Post-Stimulus)'); % Super title for the figure

%% Visualization 3: Projecting Data onto Principal Components (Scores Plot)
% The 'score' matrix contains the coordinates of your original data (each trial of each neuron)
% in the new principal component space.

% Example: Plotting PC1 vs PC2 scores for all trials of all neurons
figure;
scatter(score(:,1), score(:,2), 10, 'filled', 'MarkerFaceAlpha', 0.5); % Size 10, filled, semi-transparent
xlabel('Principal Component 1 Score');
ylabel('Principal Component 2 Score');
title('PCA Scores: Projection of Trial-Neuron PSTHs onto PC1 vs PC2');
grid on;

% If you want to color-code by neuron:
figure;
hold on;
colors = parula(nClusters); % Or any colormap
for idx = 1:nClusters
    % Get scores for trials of the current neuron
    neuronScoresPC1 = score((idx-1)*numTrials+1 : idx*numTrials, 1);
    neuronScoresPC2 = score((idx-1)*numTrials+1 : idx*numTrials, 2);
    scatter(neuronScoresPC1, neuronScoresPC2, 20, colors(idx,:), 'filled', 'MarkerFaceAlpha', 0.6, 'DisplayName', ['Neuron ' num2str(idx)]);
end
hold off;
xlabel('Principal Component 1 Score');
ylabel('Principal Component 2 Score');
title('PCA Scores (PC1 vs PC2) by Neuron - Post-Stimulus');
grid on;
legend show;












%% For each region do PCA of overlapped counts in half-second bins over full session - 20 sec to run
% overlapped bins are [0 .5], [.25 .75], [.5 1], [.75 1.25], ...
K = 10; % how many PCs to capture
vv = zeros(3,regions.N-1); % relative variances on 3 PCs
myLoadings = cell( regions.N-1,1);
timeEdges1 = 0:.5:sessionTime; timeEdges2 = 0.25:.5:sessionTime; % overlapping bins of half-second
if length(timeEdges2)==length(timeEdges1), timeEdges2 = timeEdges2(1:end-1); end % ensure interleaved second interval set ends before first set of 
% make interleaved subsets of indices within dat to put counts into
Nbins = length(timeEdges1)+length(timeEdges2)-2; % always odd
mm1 = 1:2:Nbins ; mm2 = 2:2:Nbins-1 ; 
% mm1 are odd indices, including last one; mm2 are even indices interspersed 
time2binScale = 4; % factor from time stamps to bin indices
% time = bin/4 ; time at mid-bin;  bin = round(4 * time)
PCbinTimes = (1:Nbins)/time2binScale; % times at center of bins
allScores = zeros(Nbins, 3, regions.N-1);
% Construct counts and PCs
for rr = 1:regions.N-1
    nn = find( neurons.region == rr); % get indices of neurons in region 
    myCounts = zeros( Nbins,length(nn)); % set up matrix to hold counts
    for kk = 1:length(nn)
        isNeuron = S.spikes.clusters == neurons.id(nn(kk)); % logical vector to select spikes corresponding to this neuron
        myCounts(mm1,kk) = histcounts(S.spikes.times(isNeuron),timeEdges1); % complete set of bins
        myCounts(mm2,kk) = histcounts(S.spikes.times(isNeuron),timeEdges2); % overlapping bins
    end
    [coefs, scores, ~, ~, explained ] = pca(sqrt(myCounts)); % square root of counts to keep variance equal (1/4) across variables
    allScores( :, :, rr) = scores(:, 1:3);
    vv(:,rr) = explained(1:3)/100; % variance explained
    myLoadings{rr} = coefs(:,1:3);
end

%% Now make plots showing PC scores for each region during a rest period
T1 = 0; T2 = 5000; titlestr = [num2str(T1),'s to ',num2str(T2),'s'];
T1 = 1200; T2 = 1300; titlestr = [num2str(T1),'s to ',num2str(T2),'s'];
T1 = 1300; T2 = 1400; titlestr = [num2str(T1),'s to ',num2str(T2),'s'];
nntt = find( PCbinTimes > T1 & PCbinTimes < T2) ; myTS = PCbinTimes(nntt); numCCbins = length(nntt);% indices and time stamps within interval
figure
tempList = [1,2,3,10];
for rr = 1:4
    
    subplot(2,2,rr) ; hold on
    tmp = squeeze( allScores(nntt,:,tempList(rr))) ; % PCs of this region
    for jj = 1:3, tmp(:,jj) = smooth(tmp(:,jj),'sgolay'); end
    plot3(tmp(:,1),tmp(:,2),tmp(:,3),'.')
     % add annotations - wheel moves
    LL = find( S.wheelMoves.intervals(:,1) > T1 & S.wheelMoves.intervals(:,2) < T2); % wheel interval overlaps
    for ll = 1:length(LL)   
        nntt2 = find( myTS > S.wheelMoves.intervals(LL(ll),1) & myTS < S.wheelMoves.intervals(LL(ll),2));
        plot3(tmp(nntt2,1),tmp(nntt2,2),tmp(nntt2,3),'color',[.7 .5 .4]);
        xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
        view(137,11); %title(titlestr); axis square;
    end
    title(regions.name(tempList(rr))); grid
end
