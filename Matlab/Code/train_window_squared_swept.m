%%
clc; clear; close all;

resize_dataset = true;

if resize_dataset == true 
    load ds_window_swept_new.mat
    signals_square = zeros(size(signals,1)/16, size(signals,2)*16, 1, size(signals,4));
    for i = 1 : size(signals,4)
        for m = 1 : 16 
            signals_square(:,(1+8*(m-1)):(8*m),1,i) = signals ((1+256*(m-1):256*(m)),:,1,i);
        end
    end
    save ("ds_window_swept_new_squared.mat", "signals_square", "doa", "jam", "y_ds_cell", "-v7.3","-nocompression")
else
    load ds_window_swept_new_squared.mat
end
%% CUSTOM SETTINGS
Train_perc = 0.8; %percentage of dataset used for training
Val_perc = 0.1; %percentage of dataset used for validation
Test_perc = 0.1; %percentage of dataset used for test

%% REMOVE "NO JAMMING" - EVENTUALMENTE DA RIMUOVERE
% signals=signals(:,:,:,126:end);
% doa=doa(126:end);
% jam=jam(126:end);

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
%load .\networks\db4ra_resnet18_256x128.mat
load .\networks\resnet18_nob5_256x128.mat

%% Training preparation
miniBatchSize  = 64;
validationFrequency = floor(numel(doa_Train)/miniBatchSize);
LearnRateDropPeriod = 8;
MaxEpochs = LearnRateDropPeriod*4-1;

options = trainingOptions("rmsprop", ...
    MiniBatchSize=miniBatchSize, ...
    MaxEpochs=MaxEpochs,...
    LearnRateSchedule="piecewise", ...
    InitialLearnRate=1e-3, ...
    ... %LearnRateDropFactor=0.05, ...
    LearnRateDropPeriod=LearnRateDropPeriod, ...
    Shuffle="every-epoch", ...
    ValidationData={signals_Val,[doa_Val jam_Val]}, ...
    ValidationFrequency=validationFrequency, ...
    Plots="training-progress", ...
    Metrics="rmse", ...
    Verbose=true);

    %ExecutionEnvironment="parallel", opzione che crea problemi con la ram
%% TRAINING
net = trainnet(signals_Train,[doa_Train jam_Train],net,"mse",options);

%%
save (".\networks\trained\db4ra_resnet18_256x128_swept_new.mat","net");

%% TESTING
YTest = predict(net,signals_Test);

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
hold off
figure
C = confusionmat(sign(jam_Test), double(sign(YTest(:,2))));
confusionchart(C);

%%
exportNetworkToTensorFlow(net,"db4ara_resnet_window_squared_swept")


%% TESTING
samples_with_jammer = jam_Test ==1;
figure
scatter(YTest((samples_with_jammer),1),doa_Test(samples_with_jammer),"+")
xlabel("Predicted Value")
ylabel("True Value")

hold on
plot([-81 81]/10, [-81 81]/10,"r--")
grid on
grid minor

RMSE_test_doa=rmse(YTest((samples_with_jammer),1),doa_Test(samples_with_jammer))

%%
save (".\networks\trained\test_vectors.mat","jam_Test", "signals_Test", "doa_Test");