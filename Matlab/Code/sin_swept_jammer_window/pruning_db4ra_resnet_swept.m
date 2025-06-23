
%%

addpath(fullfile(pwd, '../../', 'Lib'));  %load trainingPartitions function

%%
% IL SEGUENTE SCRIPT LAVORA SU UNA RETE CHE HA COME INGRESSO UNA MATRICE
% 512 X 64, SU UN DATASET DI 105'000 SAMPLE. IL DATASET E' COMPOSTO DA
% 30'000 CAMPIONI SENZA INTERFERENTE E 75'000 CON INTERFERENTE (1'500 FREQUENZE * 50
% ANGOLI)
% rispetto alla rete dello script train window, qui è stata ridotta la
% dimensione ddei filtri del primo strato (64x8 --> 32x8) ed è stato
% impostato lo stride a [16,8]
%cambiato il primo stride di pooling a dimensione [2,2]
% la dimensione della rete è stata sensibilmente ridotta rimuovendo il
% banch numero 5

%IL DATASET DI RIFERIMENTO E' GENERATO DALLO SCRIPT train_window_squared

clc; clear; close all;

load .\..\database\ds_window_swept_squared.mat

%% CUSTOM SETTINGS
Train_perc = 0.8; %percentage of dataset used for training
Val_perc = 0.1; %percentage of dataset used for validation
Test_perc = 0.1; %percentage of dataset used for test

%% Dataset preparation

numObservations = size(signals_square,4);

[idxTrain,idxVal,idxTest] = trainingPartitions(numObservations, [Train_perc Val_perc Test_perc]);

signals_Train = signals_square(:,:,:,idxTrain);
doa_Train = doa(idxTrain);
jam_Train = jam(idxTrain);

signals_Val = signals_square(:,:,:,idxVal);
doa_Val = doa(idxVal);
jam_Val = jam(idxVal);

signals_Test = signals_square(:,:,:,idxTest);
doa_Test = doa(idxTest);
jam_Test = jam(idxTest);

clear signals signals_square jam doa
%%
load .\..\networks\trained\db4ra_resnet18_256x128_swept.mat

%%
YTest = predict(net,signals_Test);
accuracyOfTrainedNet = rmse(YTest(:,1),doa_Test);

%%
maxPruningIterations = 30;
maxToPrune = 55;

learnRate = 1e-2;
momentum = 0.9;
miniBatchSize = 250;
numMinibatchUpdates  = 50;
validationFrequency = 1;

prunableNet = taylorPrunableNetwork(net);
maxPrunableFilters = prunableNet.NumPrunables;

figure("Position",[10,10,700,700])
tl = tiledlayout(3,1);
lossAx = nexttile;
lineLossFinetune = animatedline(Color=[0.85 0.325 0.098]);
%ylim([0 inf])
xlabel("Fine-Tuning Iteration")
ylabel("Loss")
grid on
title("Mini-Batch Loss During Pruning")
xTickPos = [];

accuracyAx = nexttile;
lineAccuracyPruning = animatedline(Color=[0.098 0.325 0.85],LineWidth=2,Marker="o");
%ylim([50 100])
xlabel("Pruning Iteration")
ylabel("Accuracy")
grid on
addpoints(lineAccuracyPruning,0,accuracyOfTrainedNet)
title("Validation Accuracy After Pruning")

numPrunablesAx = nexttile;
lineNumPrunables = animatedline(Color=[0.4660 0.6740 0.1880],LineWidth=2,Marker="^");
%ylim([200 700])
xlabel("Pruning Iteration")
ylabel("Prunable Filters")
grid on
addpoints(lineNumPrunables,0,maxPrunableFilters)
title("Number of Prunable Convolution Filters After Pruning")

%%

start = tic;
iteration = 0;

n_samples = length(doa_Train);

prunedNetsArray = cell(1,30);

for pruningIteration = 1:maxPruningIterations

    % Shuffle data.
    %shuffle(mbqTrain);

    % Reset the velocity parameter for the SGDM solver in every pruning
    % iteration.
    velocity = [];

    % Loop over mini-batches.
    fineTuningIteration = 0;
    for p = 1 : n_samples / miniBatchSize
    %while hasdata(mbqTrain)
        iteration = iteration + 1;
        fineTuningIteration = fineTuningIteration + 1;

        % Read mini-batch of data.
        %[X, T] = next(mbqTrain);
        X  = signals_Train(:,:,:,1+(p-1)*miniBatchSize:p*miniBatchSize);
        X = dlarray(X, "SSCB");
        T  = [doa_Train(1+(p-1)*miniBatchSize:p*miniBatchSize) jam_Train(1+(p-1)*miniBatchSize:p*miniBatchSize)];
        T = T';
        T = dlarray(T,"CB");
        % Evaluate the pruning activations, gradients of the pruning
        % activations, model gradients, state, and loss using the dlfeval and
        % modelLossPruning functions.
        [loss,pruningActivations, pruningGradients, netGradients, state] = ...
            dlfeval(@modelLossPruning, prunableNet, X, T);

        % Update the network state.
        prunableNet.State = state;

        % Update the network parameters using the SGDM optimizer.
        [prunableNet, velocity] = sgdmupdate(prunableNet, netGradients, velocity, learnRate, momentum);

        % Compute first-order Taylor scores and accumulate the score across
        % previous mini-batches of data.
        prunableNet = updateScore(prunableNet, pruningActivations, pruningGradients);

        % Display the training progress.
        D = duration(0,0,toc(start),Format="hh:mm:ss");
        addpoints(lineLossFinetune, iteration, loss)
        title(tl,"Processing Pruning Iteration: " + pruningIteration + " of " + maxPruningIterations + ...
            ", Elapsed Time: " + string(D))
        % Synchronize the x-axis of the accuracy and numPrunables plots with the loss plot.
        xlim(accuracyAx,lossAx.XLim)
        xlim(numPrunablesAx,lossAx.XLim)
        drawnow

        % Stop the fine-tuning loop when numMinibatchUpdates is reached.
        if (fineTuningIteration > numMinibatchUpdates)
            break
        end
    end

    % Prune filters based on previously computed Taylor scores.
    prunableNet = updatePrunables(prunableNet, MaxToPrune = maxToPrune);

    % Show results on the validation data set in a subset of pruning iterations.
    isLastPruningIteration = pruningIteration == maxPruningIterations;
    if (mod(pruningIteration, validationFrequency) == 0 || isLastPruningIteration)
        
        %accuracy = modelAccuracy(prunableNet, mbqTest, classes, augimdsTest.NumObservations);
        
        YTest = predict(prunableNet,signals_Test);
        accuracy = rmse(YTest(:,1),doa_Test);

        addpoints(lineAccuracyPruning, iteration, accuracy)
        addpoints(lineNumPrunables,iteration,prunableNet.NumPrunables)
    end
    prunedNetsArray{pruningIteration} = prunableNet;
    % Set x-axis tick values at the end of each pruning iteration.
    xTickPos = [xTickPos, iteration]; %#ok<AGROW>
    xticks(lossAx,xTickPos)
    xticks(accuracyAx,[0,xTickPos])
    xticks(numPrunablesAx,[0,xTickPos])
    xticklabels(accuracyAx,["Unpruned",string(1:pruningIteration)])
    xticklabels(numPrunablesAx,["Unpruned",string(1:pruningIteration)])
    drawnow
end

%%
save (".\..\networks\trained\db4ra_resnet18_256x128_swept_pruned_array.mat","prunedNetsArray");

%%
net_sel = 0;

switch net_sel
    case 0
        prunedNet = dlnetwork(prunedNetsArray{30});
    case 1
        load  (".\..\networks\db4ra_resnet18_256x128_swept_pruned_8x8.mat","prunedNet");
    otherwise
        disp('other value')
end


%% Training preparation
miniBatchSizeTrain  = 64;
validationFrequencyTrain = floor(numel(doa_Train)/miniBatchSizeTrain);

LearnRateDropPeriod = 5;
MaxEpochs = LearnRateDropPeriod * 3 -1;

options = trainingOptions("rmsprop", ...
    MiniBatchSize=miniBatchSizeTrain, ...
    MaxEpochs=MaxEpochs,...
    LearnRateSchedule="piecewise", ...
    InitialLearnRate=1e-3, ...
    ... %LearnRateDropFactor=0.05, ...
    LearnRateDropPeriod=LearnRateDropPeriod, ...
    Shuffle="every-epoch", ...
    ValidationData={signals_Val,[doa_Val jam_Val]}, ...
    ValidationFrequency=validationFrequencyTrain, ...
    Plots="training-progress", ...
    Metrics="rmse", ...
    Verbose=true);



prunedNetTrained = trainnet(signals_Train,[doa_Train jam_Train],prunedNet,"mse",options);

%%
switch net_sel
    case 0
        save (".\..\networks\trained\db4ra_resnet18_256x128_swept_pruned.mat","prunedNetTrained");
    case 1
        save (".\..\networks\trained\db4ra_resnet18_256x128_swept_pruned_8x8.mat","prunedNetTrained");
    otherwise
        disp('other value')
end
%%
exportNetworkToTensorFlow(prunedNetTrained,"db4ara_resnet18_nob5_256x128_pruned")

%% TESTING
YTest = predict(prunedNetTrained,signals_Test);

figure
scatter(YTest(:,1),doa_Test,"+")
xlabel("Predicted Value")
ylabel("True Value")

hold on
plot([-81 81]/10, [-81 81]/10,"r--")
grid on
grid minor

RMSE_test_doa=rmse(YTest(:,1),doa_Test)

figure
scatter(sign(YTest(:,2)),sign(jam_Test),"+")
xlabel("Predicted Value")
ylabel("True Value")

hold on
grid on
grid minor

RMSE_test_jam=rmse(YTest(:,1),jam_Test)
hold off

%% TESTING
YTest = predict(net,signals_Test);

figure
scatter(YTest(:,1),doa_Test,"+")
xlabel("Predicted Value")
ylabel("True Value")

hold on
plot([-81 81]/10, [-81 81]/10,"r--")
grid on
grid minor

RMSE_test_doa=rmse(YTest(:,1),doa_Test)

figure
scatter(sign(YTest(:,2)),sign(jam_Test),"+")
xlabel("Predicted Value")
ylabel("True Value")

hold on
grid on
grid minor

RMSE_test_jam=rmse(YTest(:,1),jam_Test)
hold off
%%

C = confusionmat(sign(jam_Test), double(sign(YTest(:,2))));
confusionchart(C);

%%
[originalNetFilters,layerNames] = numConvLayerFilters(net);
prunedNetFilters = numConvLayerFilters(prunedNetTrained);

figure("Position",[10,10,900,900])
bar([originalNetFilters,prunedNetFilters])
xlabel("Layer")
ylabel("Number of Filters")
title("Number of Filters per Layer")
xticks(1:(numel(layerNames)))
xticklabels(layerNames)
xtickangle(90)
ax = gca;
ax.TickLabelInterpreter = "none";
legend("Original Network Filters","Pruned Network Filters","Location","southoutside")

%%
pruned_filters_array = zeros(15,length(prunedNetsArray));
for i = 1 : length(prunedNetsArray)
    [pruned_filters_array(:,i),layerNames] = numConvLayerFilters(dlnetwork(prunedNetsArray{i}));
end
%%
figure
hold on
plot(pruned_filters_array(1,:))
plot(pruned_filters_array(2,:))
plot(pruned_filters_array(3,:))
plot(pruned_filters_array(4,:))
plot(pruned_filters_array(5,:))
legend
hold off

figure
hold on
plot(pruned_filters_array(6,:))
plot(pruned_filters_array(7,:))
plot(pruned_filters_array(8,:))
plot(pruned_filters_array(9,:))
plot(pruned_filters_array(10,:))
legend
hold off

figure
hold on
plot(pruned_filters_array(11,:))
plot(pruned_filters_array(12,:))
plot(pruned_filters_array(13,:))
plot(pruned_filters_array(14,:))
plot(pruned_filters_array(15,:))
legend
hold off

%%
switch net_sel
    case 0
        load  (".\..\networks\trained\db4ra_resnet18_256x128_swept_pruned.mat","prunedNetTrained");
    case 1
        load  (".\..\networks\trained\db4ra_resnet18_256x128_swept_pruned_8x8.mat","prunedNetTrained");
    otherwise
        disp('other value')
end

net = prunedNetTrained;

YTest = predict(net,signals_Test);
accuracyOfTrainedNet = rmse(YTest(:,1),doa_Test);

%%
maxPruningIterations = 30;
maxToPrune = 15;

learnRate = 1e-2;
momentum = 0.9;
miniBatchSize = 250;
numMinibatchUpdates  = 50;
validationFrequency = 1;

prunableNet = taylorPrunableNetwork(net);
maxPrunableFilters = prunableNet.NumPrunables;

figure("Position",[10,10,700,700])
tl = tiledlayout(3,1);
lossAx = nexttile;
lineLossFinetune = animatedline(Color=[0.85 0.325 0.098]);
%ylim([0 inf])
xlabel("Fine-Tuning Iteration")
ylabel("Loss")
grid on
title("Mini-Batch Loss During Pruning")
xTickPos = [];

accuracyAx = nexttile;
lineAccuracyPruning = animatedline(Color=[0.098 0.325 0.85],LineWidth=2,Marker="o");
%ylim([50 100])
xlabel("Pruning Iteration")
ylabel("Accuracy")
grid on
addpoints(lineAccuracyPruning,0,accuracyOfTrainedNet)
title("Validation Accuracy After Pruning")

numPrunablesAx = nexttile;
lineNumPrunables = animatedline(Color=[0.4660 0.6740 0.1880],LineWidth=2,Marker="^");
%ylim([200 700])
xlabel("Pruning Iteration")
ylabel("Prunable Filters")
grid on
addpoints(lineNumPrunables,0,maxPrunableFilters)
title("Number of Prunable Convolution Filters After Pruning")

%%

start = tic;
iteration = 0;

n_samples = length(doa_Train);

prunedNetsArrayFine = cell(1,30);

for pruningIteration = 1:maxPruningIterations

    % Shuffle data.
    %shuffle(mbqTrain);

    % Reset the velocity parameter for the SGDM solver in every pruning
    % iteration.
    velocity = [];

    % Loop over mini-batches.
    fineTuningIteration = 0;
    for p = 1 : n_samples / miniBatchSize
    %while hasdata(mbqTrain)
        iteration = iteration + 1;
        fineTuningIteration = fineTuningIteration + 1;

        % Read mini-batch of data.
        %[X, T] = next(mbqTrain);
        X  = signals_Train(:,:,:,1+(p-1)*miniBatchSize:p*miniBatchSize);
        X = dlarray(X, "SSCB");
        T  = [doa_Train(1+(p-1)*miniBatchSize:p*miniBatchSize) jam_Train(1+(p-1)*miniBatchSize:p*miniBatchSize)];
        T = T';
        T = dlarray(T,"CB");
        % Evaluate the pruning activations, gradients of the pruning
        % activations, model gradients, state, and loss using the dlfeval and
        % modelLossPruning functions.
        [loss,pruningActivations, pruningGradients, netGradients, state] = ...
            dlfeval(@modelLossPruning, prunableNet, X, T);

        % Update the network state.
        prunableNet.State = state;

        % Update the network parameters using the SGDM optimizer.
        [prunableNet, velocity] = sgdmupdate(prunableNet, netGradients, velocity, learnRate, momentum);

        % Compute first-order Taylor scores and accumulate the score across
        % previous mini-batches of data.
        prunableNet = updateScore(prunableNet, pruningActivations, pruningGradients);

        % Display the training progress.
        D = duration(0,0,toc(start),Format="hh:mm:ss");
        addpoints(lineLossFinetune, iteration, loss)
        title(tl,"Processing Pruning Iteration: " + pruningIteration + " of " + maxPruningIterations + ...
            ", Elapsed Time: " + string(D))
        % Synchronize the x-axis of the accuracy and numPrunables plots with the loss plot.
        xlim(accuracyAx,lossAx.XLim)
        xlim(numPrunablesAx,lossAx.XLim)
        drawnow

        % Stop the fine-tuning loop when numMinibatchUpdates is reached.
        if (fineTuningIteration > numMinibatchUpdates)
            break
        end
    end

    % Prune filters based on previously computed Taylor scores.
    prunableNet = updatePrunables(prunableNet, MaxToPrune = maxToPrune);

    % Show results on the validation data set in a subset of pruning iterations.
    isLastPruningIteration = pruningIteration == maxPruningIterations;
    if (mod(pruningIteration, validationFrequency) == 0 || isLastPruningIteration)
        
        %accuracy = modelAccuracy(prunableNet, mbqTest, classes, augimdsTest.NumObservations);
        
        YTest = predict(prunableNet,signals_Test);
        accuracy = rmse(YTest(:,1),doa_Test);

        addpoints(lineAccuracyPruning, iteration, accuracy)
        addpoints(lineNumPrunables,iteration,prunableNet.NumPrunables)
    end
    prunedNetsArrayFine{pruningIteration} = prunableNet;
    % Set x-axis tick values at the end of each pruning iteration.
    xTickPos = [xTickPos, iteration]; %#ok<AGROW>
    xticks(lossAx,xTickPos)
    xticks(accuracyAx,[0,xTickPos])
    xticks(numPrunablesAx,[0,xTickPos])
    xticklabels(accuracyAx,["Unpruned",string(1:pruningIteration)])
    xticklabels(numPrunablesAx,["Unpruned",string(1:pruningIteration)])
    drawnow
end

%%

switch net_sel
    case 0
        save (".\..\networks\trained\swept_resnet18_nob5_256x128_pruned_array_fine.mat","prunedNetsArrayFine");
    case 1
        save (".\..\networks\trained\swept_resnet18_nob5_256x128_pruned_array_fine_8x8.mat","prunedNetsArrayFine");
    otherwise
        disp('other value')
end

%%
prunedNetFine = dlnetwork(prunedNetsArrayFine{20});

%% Training preparation
miniBatchSizeTrain  = 64;
validationFrequencyTrain = floor(numel(doa_Train)/miniBatchSizeTrain);

LearnRateDropPeriod = 5;
MaxEpochs = LearnRateDropPeriod * 3 -1;

options = trainingOptions("rmsprop", ...
    MiniBatchSize=miniBatchSizeTrain, ...
    MaxEpochs=MaxEpochs,...
    LearnRateSchedule="piecewise", ...
    InitialLearnRate=1e-3, ...
    ... %LearnRateDropFactor=0.05, ...
    LearnRateDropPeriod=LearnRateDropPeriod, ...
    Shuffle="every-epoch", ...
    ValidationData={signals_Val,[doa_Val jam_Val]}, ...
    ValidationFrequency=validationFrequencyTrain, ...
    Plots="training-progress", ...
    Metrics="rmse", ...
    Verbose=true);


prunedNetFineTrained = trainnet(signals_Train,[doa_Train jam_Train],prunedNetFine,"mse",options);

%%
save (".\..\networks\trained\db4ra_resnet18_256x128_swept_pruned_fine_20_8x8.mat","prunedNetFineTrained");

%%
[originalNetFilters,layerNames] = numConvLayerFilters(net);
prunedNetFilters = numConvLayerFilters((prunedNetFineTrained));

figure("Position",[10,10,900,900])
bar([originalNetFilters,prunedNetFilters])
xlabel("Layer")
ylabel("Number of Filters")
title("Number of Filters per Layer")
xticks(1:(numel(layerNames)))
xticklabels(layerNames)
xtickangle(90)
ax = gca;
ax.TickLabelInterpreter = "none";
legend("Original Network Filters","Pruned Network Filters","Location","southoutside")

%%
exportNetworkToTensorFlow(prunedNetFineTrained,"db4ara_resnet_swept_pruned_fine")

%%

%qYTest = predict(quantizedNet,signals_Test);
fpYTest = predict(prunedNetFineTrained,signals_Test);
%%

figure
scatter(fpYTest(:,1),doa_Test,"+")
xlabel("Predicted Value")
ylabel("True Value")

hold on
plot([-81 81]/10, [-81 81]/10,"r--")
grid on
grid minor
hold off
fpRMSE_test_doa=rmse(fpYTest(:,1),doa_Test)

%%
figure
scatter(qYTest(:,1),doa_Test,"+")
xlabel("Predicted Value")
ylabel("True Value")

hold on
plot([-81 81]/10, [-81 81]/10,"r--")
grid on
grid minor
hold off
qRMSE_test_doa=rmse(qYTest(:,1),doa_Test)