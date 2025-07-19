clear all; close all; clc

%% set ALL your settings here:
% Session(s) and Brain area(s):
    sesPath = '../data/Steinmetz/Cori_2016-12-18'; % sample with both motor and sensory areas
    areaID = 10; % hard-coded index for area VISp at the moment (once we are ready to include a second brain structure, Nick can write a quick function that looks for the string of the desired brain area(s) and returns the correct index number to select it)

% PSTH parameters:
    selectedEvents = 'stimTimes'; %event you want the PSTH to be centered on. (make sure to input it as a string)
    win = [-0.5,0.5]; % window for psth in seconds: [seconds before event, seconds after event]
    binSize = 0.02; % bin size for psth in seconds
    numOfIndividualPSTHsToPlot = 0; % number of cells you want to make individual plots for (ex. set this to 20 to plot the first 20 cells)

%% Load data and declare variables:
% load data:
    [S, regions, neurons, trials] = stOpenSession(sesPath);  % load .npy files in which data stored; % Note that regions are indexed common style from 1 to regions.N, but neurons are indexed Python-style from 0 to neurons.N-1
% declare variables:    
    clusterIDs = find(neurons.region==areaID); % find the clusterIDs specific to the selected brain area
    sessionTime = S.spikes.times(end); % total time, assuming start at 0
    stimTimes = trials.visStimTime; 
    respTimes = trials.responseTime;
    goTimes = S.trials.goCue_times;
        [selectedEvents] = SelectEventVariableFromString(selectedEvents,stimTimes,respTimes,goTimes); %produces the times of your desired event (function located at bottom of script)

figure; 
plot(stimTimes,ones(size(stimTimes)),'.'); hold on;
plot(respTimes,ones(size(stimTimes)),'r.')
grid on; grid minor; xlabel('seconds')

%minimum of 0.5s response after stimulus presentation
%Inter trial interval minimum of 1 second

% not used at the moment, but probably the best way to select selected trial types 
% % % construct logical variable for spike timestamps in trials
% % % inTrial = false(size(S.spikes.times,1),1);
% % % for kk = 1:trials.N
% % %     inTrial( S.spikes.times > stimTimes(kk) & S.spikes.times < respTimes(kk) ) = true;
% % % end

%% (not used yet, but will be useful) Parameters to convert between frame numbers for behavior and time in seconds
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

%% create structure for the spikes we want to analyze 
    % when we incorporate multiple brain areas, we can refactor this to produce a spike structure for each brain area. 
    % alternativly, we could instead introduce a saving function at the end of the script, such that we could load up the processed data in another script later which analyzed multiple brain areas
nClusters = length(clusterIDs);
for idx = 1:nClusters
    spikes(idx).clu = (clusterIDs(idx));
    spikes(idx).spikeIndex = find(S.spikes.clusters(:)==spikes(idx).clu);
    spikes(idx).spiketimes = S.spikes.times(spikes(idx).spikeIndex);
    
    %organize into matrix around stimTime:
    [spikes(idx).psth, bins, spikes(idx).rasterX, spikes(idx).rasterY, spikes(idx).spikeCounts, spikes(idx).binnedArray] = psthAndBA(spikes(idx).spiketimes, selectedEvents, win, binSize);
end

%% plot individual cell's PSTH for first numOfIndividualPSTHsToPlot
if gt(numOfIndividualPSTHsToPlot,0)
    sm =  0.25; colors = turbo(100);
    for idx = 1:numOfIndividualPSTHsToPlot
        figure; tiledlayout(2,1); nexttile; axR = gca; nexttile; axP = gca;
        rasterAndPSTHbyCond(spikes(idx).spiketimes, stimTimes, trials.contrast, win, sm, colors, axR, axP)
    end
end
%%

% % timeBinCenters = bins(1:end-1) + diff(bins)/2;
% % startPostStimBinIdx = find(timeBinCenters >= 0, 1, 'first');
% % if isempty(startPostStimBinIdx)
% %     warning('Could not find a bin center at or after 0. Check your win/binSize and bins calculation.');
% %     startPostStimBinIdx = 1; % Fallback to start from beginning if issue
% % end
% % endPostStimBinIdx = length(timeBinCenters); % End of the time series
% % 
% % % This is the range of time bins for your post-stimulus activity
% % postStimBinIndices = startPostStimBinIdx : endPostStimBinIdx;
% % numPostStimBins = length(postStimBinIndices); % This will dynamically be 25 in your case

%% Prepare data for PCA
numTrials = size(spikes(1).binnedArray, 1);
numTotalTimeBins = size(spikes(1).psth, 2); % This will be 50

% Initialize the matrix for PCA with the correct number of columns
% neuralDataForPCA = zeros(numTrials * nClusters, length(bins));
neuralDataForPCA = zeros(nClusters, length(bins));
neuralDataForPCAnew = zeros(nClusters*numTrials,length(bins));
k = 0
for idx = 1:nClusters
    % Each rows corresponds to a single neuron's PSTH across all trials
    neuralDataForPCA(idx, :) = spikes(idx).psth;
    num = ((idx-1)*numTrials)+1;
    neuralDataForPCAnew(num:num+numTrials-1, :) = spikes(idx).binnedArray/0.02;
end

%% Run PCA Original
% fix: make it divide matrix by bin size
% neuralDataForPCA = sqrt(neuralDataForPCA);
[eigenvectors_PCA, proj_PCA, eigenvalues_PCA, tsquared, explained, mu] = pca(neuralDataForPCA); % here are the default output names: [coeff, score, latent, tsquared, explained, mu] = pca(neuralDataForPCA);
    n = size(neuralDataForPCA', 1); % number of observations
    singularValues = sqrt(eigenvalues_PCA * (n - 1));
    totalVariance = sum(singularValues.^2);
    percentVariance = (singularValues.^2 / totalVariance) * 100;
%% Plot PCA
% bar(percentVariance,)
tiledlayout(1,2);
nexttile(1); bar(bins,abs(eigenvectors_PCA),'stacked'); ylabel('component size'); xlabel('s');
nexttile(2); bar(bins,abs(eigenvectors_PCA(:,1:5)'),'stacked'); ylabel('component size'); xlabel('s');

%% Running SVD as well (to confirm I am not fumbling with the PCA)
[U,Smat,V] = svd(neuralDataForPCA,"econ");
    Smat = diag(Smat); %singular values
figure; bar(1:length(Smat),Smat); %scree plot
    title('SVD scree plot'); ylabel('Singular Value'); xlabel('Component Number'); %xlim([0,10]); %uncomment to view just first 10
figure; bar(bins,abs(V),'stacked'); %plot each component as raw eigenvalues
    title("Components' explanatory power across the trial"); ylabel('component size'); xlabel('s');
figure; bar(bins,abs(V).*Smat','stacked'); %plot each component weighted by its singular value
    title("Weighted components' explanatory power across the trial"); ylabel('component size'); xlabel('s'); 


%% Run PCA NEW
% fix: make it divide matrix by bin size
% neuralDataForPCA = sqrt(neuralDataForPCA);
[eigenvectors_PCA, proj_PCA, eigenvalues_PCA, tsquared, explained, mu] = pca(neuralDataForPCAnew'); % here are the default output names: [coeff, score, latent, tsquared, explained, mu] = pca(neuralDataForPCA);
    n = size(neuralDataForPCA', 1); % number of observations
    singularValues = sqrt(eigenvalues_PCA * (n - 1));
    totalVariance = sum(singularValues.^2);
    percentVariance = (singularValues.^2 / totalVariance) * 100;
%% Plot PCA
% bar(percentVariance,)
bar(bins,abs(eigenvectors_PCA),'stacked'); ylabel('component size'); xlabel('s');
bar(bins,abs(eigenvectors_PCA(:,1:5)'),'stacked'); ylabel('component size'); xlabel('s');

%% Running SVD as well (to confirm I am not fumbling with the PCA)
[U,Smat,V] = svd(neuralDataForPCAnew',"econ");
    Smat = diag(Smat); %singular values
figure; bar(1:length(Smat),Smat); %scree plot
    title('SVD scree plot'); ylabel('Singular Value'); xlabel('Component Number'); %xlim([0,10]); %uncomment to view just first 10
figure; bar(bins,abs(V),'stacked'); %plot each component as raw eigenvalues
    title("Components' explanatory power across the trial"); ylabel('component size'); xlabel('s');
figure; bar(bins,abs(V).*Smat','stacked'); %plot each component weighted by its singular value
    title("Weighted components' explanatory power across the trial"); ylabel('component size'); xlabel('s'); 



















%% STUFF I HAVENT GOT TO / LOOKED AT YET: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


%% FUNCTIONS %%%%% functions used internally in this script
function [outputEventTimes] = SelectEventVariableFromString(stringInput,stimTimes,respTimes,goTimes)
% quick dumb function that takes in the string you declared for the events 
% you wanted to look at, and actually returns the values. This is necessary
% if you want to declare this at the beginning of the script, because the
% values (the variable) isn't defined at that point

if isequal(stringInput,'stimTimes')
    outputEventTimes = stimTimes;
elseif isequal(stringInput,'respTimes')
    outputEventTimes = respTimes;
elseif isequal(stringInput,'goTimes')
    outputEventTimes = goTimes;
end
end