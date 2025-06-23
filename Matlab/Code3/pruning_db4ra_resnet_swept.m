
%%
function [loss,pruningGradient,pruningActivations,netGradients,state] = modelLossPruning(prunableNet, X, T)

[dlYPred,state,pruningActivations] = forward(prunableNet,X);

%loss = crossentropy(dlYPred,T);
loss = mse(dlYPred,T);
[pruningGradient,netGradients] = dlgradient(loss,pruningActivations,prunableNet.Learnables);

end
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

load ds_window_swept_squared2.mat

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
load .\networks\trained\db4ra_resnet18_256x128_swept.mat

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
save (".\networks\db4ra_resnet18_256x128_swept_pruned_array.mat","prunedNetsArray");

%%
net_sel = 2;

switch net_sel
    case 0
        prunedNet = dlnetwork(prunedNetsArray{26});
    case 1
        load (".\networks\db4ra_resnet18_256x128_swept_pruned_8x8.mat","prunedNet");
    case 2
        load (".\networks\db4ra_resnet18_256x128_swept_pruned_26_8x8.mat","prunedNet");
    otherwise
        disp('other value')
end


%% Training preparation
miniBatchSizeTrain  = 64;
validationFrequencyTrain = floor(numel(doa_Train)/miniBatchSizeTrain);

LearnRateDropPeriod = 6;
MaxEpochs = LearnRateDropPeriod * 4 -1;

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
        save (".\networks\trained\db4ra_resnet18_256x128_swept_pruned.mat","prunedNetTrained");
        exportNetworkToTensorFlow(prunedNetTrained,"db4ara_resnet18_nob5_256x128_pruned")
    case 1
        save (".\networks\trained\db4ra_resnet18_256x128_swept_pruned_8x8.mat","prunedNetTrained");
        exportNetworkToTensorFlow(prunedNetTrained,"db4ara_resnet18_nob5_256x128_pruned_8x8")
    case 2
        save (".\networks\trained\db4ra_resnet18_256x128_swept_pruned_26_8x8.mat","prunedNetTrained26");
    otherwise
        disp('other value')
end

%% TESTING
YTest = predict(prunedNetTrained,signals_Test);

doa_Pred = YTest(:,1);
doa_Pred_unnorm = doa_Pred *100 / 8;
doa_Test_unnorm = doa_Test * 100 / 8;

jam_Pred = YTest(:,2);

figure
scatter(doa_Pred_unnorm,doa_Test_unnorm,"+")
xlabel("Predicted Value")
ylabel("True Value")

hold on
plot([-81 81], [-81 81],"r--")
grid on
grid minor
hold off

RMSE_test_doa=rmse(doa_Pred_unnorm,doa_Test_unnorm)

figure
scatter(sign(jam_Pred),sign(jam_Test),"+")
xlabel("Predicted Value")
ylabel("True Value")

hold on
grid on
grid minor
hold off
RMSE_test_jam=rmse(jam_Pred,jam_Test)

%%
figure
C = confusionmat(sign(jam_Test), double(sign(jam_Pred)));
confusionchart(C);

%%
function [nFilters, convNames] = numConvLayerFilters(net)
numLayers = numel(net.Layers);
convNames = [];
nFilters = [];
% Check for convolution layers and extract the number of filters.
for cnt = 1:numLayers
    if isa(net.Layers(cnt),"nnet.cnn.layer.Convolution2DLayer")
        sizeW = size(net.Layers(cnt).Weights);
        nFilters = [nFilters; sizeW(end)];
        convNames = [convNames; string(net.Layers(cnt).Name)];
    end
end
end

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

figure;

markerStyles = {'o', 's', 'd', '^', 'v'};  % cerchio, quadrato, rombo, triangolo su, triangolo giù

for i = 1:3
    subplot(3,1,i);  % 3 righe, 1 colonna, subplot i
    hold on;
    idx_start = (i-1)*5 + 1;
    idx_end = i*5;

    for j = idx_start:idx_end
        plot(pruned_filters_array(j,:), ...
            'DisplayName', strrep(layerNames(j), '_', '\_'), ...
            'Marker', markerStyles{mod(j-1,5)+1});
    end
    
    xlabel("Pruning Iteration")
    ylabel("Number of Filters")
    grid on;
    grid minor
    legend show;
    title(['Subplot Res ' num2str(i)]);
    hold off;
end

%%
switch net_sel
    case 0
        load (".\networks\trained\db4ra_resnet18_256x128_swept_pruned.mat","prunedNetTrained");
    case 1
        load (".\networks\trained\db4ra_resnet18_256x128_swept_pruned_8x8.mat","prunedNetTrained");
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
        save (".\networks\swept_resnet18_nob5_256x128_pruned_array_fine.mat","prunedNetsArrayFine");
    case 1
        save (".\networks\swept_resnet18_nob5_256x128_pruned_array_fine_8x8.mat","prunedNetsArrayFine");
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


prunedNetFineTrained20 = trainnet(signals_Train,[doa_Train jam_Train],prunedNetFine,"mse",options);

%%

switch net_sel
    case 0
        save (".\networks\trained\db4ra_resnet18_256x128_swept_pruned_fine_20.mat","prunedNetFineTrained20");
        exportNetworkToTensorFlow(prunedNetFineTrained20,"db4ara_resnet_swept_pruned_fine_20")
    case 1   
        save (".\networks\trained\db4ra_resnet18_256x128_swept_pruned_fine_20_8x8.mat","prunedNetFineTrained20");
    otherwise
        disp('other value')
end


%%
[originalNetFilters,layerNames] = numConvLayerFilters(net);
prunedNetFilters = numConvLayerFilters((prunedNetFineTrained20));

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
prunedNetFine = dlnetwork(prunedNetsArrayFine{30});

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


prunedNetFineTrained30 = trainnet(signals_Train,[doa_Train jam_Train],prunedNetFine,"mse",options);

%%

switch net_sel
    case 0
        save (".\networks\trained\db4ra_resnet18_256x128_swept_pruned_fine_30.mat","prunedNetFineTrained30");
        exportNetworkToTensorFlow(prunedNetFineTrained30,"db4ara_resnet_swept_pruned_fine_30")
    case 1   

        save (".\networks\trained\db4ra_resnet18_256x128_swept_pruned_fine_30_8x8.mat","prunedNetFineTrained30");

    otherwise
        disp('other value')
end

%%
YTestFine20 = predict(prunedNetFineTrained20,signals_Test);
YTestFine30 = predict(prunedNetFineTrained30,signals_Test);

%%
doa_PredFine20 = YTestFine20(:,1);
doa_PredFine20_unnorm = doa_PredFine20 *100 / 8;
doa_Test_unnorm = doa_Test * 100 / 8;

jam_PredFine20 = YTestFine20(:,2);

figure
scatter(doa_PredFine20_unnorm,doa_Test_unnorm,"+")
xlabel("Predicted Value")
ylabel("True Value")

hold on
plot([-81 81], [-81 81],"r--")
grid on
grid minor
hold off

RMSE_test_doa=rmse(doa_PredFine20_unnorm,doa_Test_unnorm)

figure
scatter(sign(jam_PredFine20),sign(jam_Test),"+")
xlabel("Predicted Value")
ylabel("True Value")

hold on
grid on
grid minor
hold off
RMSE_test_jam=rmse(jam_PredFine20,jam_Test)

%%
doa_PredFine30 = YTestFine30(:,1);
doa_PredFine30_unnorm = doa_PredFine30 *100 / 8;
doa_Test_unnorm = doa_Test * 100 / 8;

jam_PredFine30 = YTestFine30(:,2);

figure
scatter(doa_PredFine30_unnorm,doa_Test_unnorm,"+")
xlabel("Predicted Value")
ylabel("True Value")

hold on
plot([-81 81], [-81 81],"r--")
grid on
grid minor
hold off

RMSE_test_doa=rmse(doa_PredFine30_unnorm,doa_Test_unnorm)

figure
scatter(sign(jam_PredFine30),sign(jam_Test),"+")
xlabel("Predicted Value")
ylabel("True Value")

hold on
grid on
grid minor
hold off
RMSE_test_jam=rmse(jam_PredFine30,jam_Test)