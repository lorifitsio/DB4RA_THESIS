clc; clear; close all;

load .\..\database\ds_signal_only.mat

addpath(fullfile(pwd, '../../', 'Lib'));  %load trainingPartitions function

%% CUSTOM SETTINGS
Train_perc = 0.8; %percentage of dataset used for training
Val_perc = 0.1; %percentage of dataset used for validation
Test_perc = 0.1; %percentage of dataset used for test

%% Dataset preparation

numObservations = size(signals,4);

[idxTrain,idxVal,idxTest] = trainingPartitions(numObservations, [Train_perc Val_perc Test_perc]);

signals_Train = signals(:,:,:,idxTrain);
doa_Train = doa(idxTrain);

signals_Val = signals(:,:,:,idxVal);
doa_Val = doa(idxVal);

signals_Test = signals(:,:,:,idxTest);
doa_Test = doa(idxTest);

clear signals doa
%%
load .\..\networks\resnet_only_signal_net.mat

%% Training preparation
miniBatchSize  = 64;
validationFrequency = floor(numel(doa_Train)/miniBatchSize);


options = trainingOptions("rmsprop", ...
    MiniBatchSize=miniBatchSize, ...
    MaxEpochs=100,...
    LearnRateSchedule="piecewise", ...
    InitialLearnRate=1e-3, ...
    LearnRateDropFactor=0.05, ...
    LearnRateDropPeriod=20, ...
    Shuffle="every-epoch", ...
    ValidationData={signals_Val,doa_Val}, ...
    ValidationFrequency=validationFrequency, ...
    Plots="training-progress", ...
    Metrics="rmse", ...
    Verbose=true);

%% TRAINING
net = trainnet(signals_Train,doa_Train,net,"mse",options);

%% TESTING
YTest = predict(net,signals_Test);

doa_Pred = YTest;
RMSE_test_doa=rmse(doa_Pred,doa_Test);

figure
scatter(doa_Pred,doa_Test,"+")
xlabel("Predicted Value")
ylabel("True Value")

hold on
plot([-61 61], [-61 61],"r--")
title("RMSE DOA [°]", num2str(RMSE_test_doa))
grid on
grid minor
hold off

%%
save(".\..\networks\trained\resnet_only_signal_net.mat","net")
