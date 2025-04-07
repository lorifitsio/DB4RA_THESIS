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

resize_dataset = false;

if resize_dataset == true 
    square_factor = 8; 
    load ds_window.mat
    signals_square = zeros(size(signals,1)/square_factor, size(signals,2)*square_factor, 1, size(signals,4));
    for i = 1 : size(signals,4)
        for m = 1 : square_factor 
            signals_square(:,(1+8*(m-1)):(8*m),1,i) = signals ((1+(4096/square_factor)*(m-1):(4096/square_factor)*(m)),:,1,i);
        end
    end
    save ("ds_window_squared.mat", "signals_square", "doa", "jam", "y_ds_cell", "-v7.3","-nocompression")
else
    load ds_window_squared.mat
end

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
load .\networks\resnet18_mod0_window_squared.mat

%% Training preparation
miniBatchSize  = 64;
validationFrequency = floor(numel(doa_Train)/miniBatchSize);


options = trainingOptions("rmsprop", ...
    MiniBatchSize=miniBatchSize, ...
    MaxEpochs=35,...
    LearnRateSchedule="piecewise", ...
    InitialLearnRate=1e-3, ...
    LearnRateDropFactor=0.05, ...
    LearnRateDropPeriod=20, ...
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
save (".\networks\trained\resnet18_mod0_window_squared.mat","net");

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
%%
hold off
C = confusionmat(sign(jam_Test), double(sign(YTest(:,2))));
confusionchart(C);

%%
exportNetworkToTensorFlow(net,"db4ara_resnet_window_squared")


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