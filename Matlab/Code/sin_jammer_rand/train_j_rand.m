clc; clear; close all;

load .\..\database\ds_rand.mat

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
jam_Train = jam(idxTrain);

signals_Val = signals(:,:,:,idxVal);
doa_Val = doa(idxVal);
jam_Val = jam(idxVal);

signals_Test = signals(:,:,:,idxTest);
doa_Test = doa(idxTest);
jam_Test = jam(idxTest);

clear signals jam doa
%%
load .\..\networks\resnet18_mod4_j0.mat

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
    ValidationData={signals_Val,[doa_Val jam_Val]}, ...
    ValidationFrequency=validationFrequency, ...
    Plots="training-progress", ...
    Metrics="rmse", ...
    ExecutionEnvironment="gpu", ...
    Verbose=true);

    %ExecutionEnvironment="parallel", opzione che crea problemi con la ram

%% TRAINING
net = trainnet(signals_Train,[doa_Train jam_Train],net,"mse",options);
save (".\..\networks\trained\sin_jammer_rand_net.mat","net");

%% TESTING
YTest = predict(net,signals_Test);

figure
scatter(YTest(:,1),doa_Test,"+")
xlabel("Predicted Value")
ylabel("True Value")

hold on
plot([-81 81], [-81 81],"r--")
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



figure
C = confusionmat(sign(jam_Test), double(sign(YTest(:,2))));
CC = confusionchart(C);
CC.Title = 'Jammer Detection';
CC.RowSummary = 'row-normalized';
CC.ColumnSummary = 'column-normalized';