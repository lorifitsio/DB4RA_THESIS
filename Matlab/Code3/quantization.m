%%
%useful link with examples
%https://it.mathworks.com/help/deeplearning/ref/dlquantizer.quantize.html#mw_eebf44dc-5794-4132-a4e8-fac660e149ec

clc; clear; close all;

load ds_window_swept_squared.mat

%% CUSTOM SETTINGS
Train_perc = 0.6; %percentage of dataset used for training
Val_perc = 0.2; %percentage of dataset used for validation
Test_perc = 0.2; %percentage of dataset used for test

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
load ds_window_swept_test.mat

use_8x8_stride = true;

if use_8x8_stride == false 
    
    load .\networks\trained\db4ra_resnet18_256x128_swept.mat
    load .\networks\trained\db4ra_resnet18_256x128_swept_pruned.mat
    %load .\networks\trained\db4ra_resnet18_256x128_swept_pruned_fine_20.mat
    %load .\networks\trained\db4ra_resnet18_256x128_swept_pruned_fine_30.mat
    
    n_of_nets = 2;
    
    net_array = cell(1,n_of_nets);
    
    net_array{1} = net;
    net_array{2} = prunedNetTrained;
    %net_array{3} = prunedNetFineTrained20;
    %net_array{4} = prunedNetFineTrained30;

else
    load .\networks\trained\db4ra_resnet18_256x128_swept_pruned_8x8.mat
    load .\networks\trained\db4ra_resnet18_256x128_swept_pruned_26_8x8.mat
    %load .\networks\trained\db4ra_resnet18_256x128_swept_pruned_fine_20_8x8.mat
    %load .\networks\trained\db4ra_resnet18_256x128_swept_pruned_fine_30_8x8.mat
    
    n_of_nets = 2;
    
    net_array = cell(1,n_of_nets);
    
    net_array{1} = prunedNetTrained;
    net_array{2} = prunedNetTrained26;
    %net_array{3} = prunedNetFineTrained30;

end
quantObj_array = cell(1,n_of_nets);
qNet_array = cell(1,n_of_nets);
qDetails_array = cell(1,n_of_nets);

%% TESTING


for i = 1 : n_of_nets 
    quantObj_array{i} = dlquantizer(net_array{i},'ExecutionEnvironment','FPGA');
    calibrate(quantObj_array{i},signals_Val);
    qNet_array{i} = quantize (quantObj_array{i}, 'ExponentScheme','Histogram');
    qDetails_array{i} = quantizationDetails(qNet_array{i});
end


%% TESTING

YTest = zeros(size(doa_Test,1),2,n_of_nets);

for i = 1 : n_of_nets 
    YTest(:,:,i) = predict(qNet_array{i},signals_Test);
end

%% TESTING
doa_Pred = YTest(:,1,:);
doa_Pred_denorm = doa_Pred *100 / 8;
doa_Test_unnorm = doa_Test * 100 / 8;

jam_Pred = YTest(:,2,:);

for i = 1 : n_of_nets
    
    figure
    scatter(doa_Pred_denorm(:,i),doa_Test_unnorm,"+")
    xlabel("Predicted Value")
    ylabel("True Value")
    
    hold on
    plot([-81 81], [-81 81],"r--")
    grid on
    grid minor
    hold off
    
    RMSE_test_doa(i)=rmse(doa_Pred_denorm(:,i),doa_Test_unnorm)
    
    figure
    scatter(sign(jam_Pred(:,i)),sign(jam_Test),"+")
    xlabel("Predicted Value")
    ylabel("True Value")
    
    hold on
    grid on
    grid minor
    hold off
    RMSE_test_jam(:,i)=rmse(jam_Pred(:,i),jam_Test)
    
    figure
    C = confusionmat(sign(jam_Test), double(sign(jam_Pred(:,i))));
    CC = confusionchart(C);
    CC.Title = 'Jammer Detection';
    CC.RowSummary = 'row-normalized';
    CC.ColumnSummary = 'column-normalized';
end


%%
%result floating point network
%RMSE_test_doa =[    0.5377    0.7465    0.7229    1.7485];

%results of quantized network
%RMSE_test_doa =[ 0.7393    0.7046    0.8862    1.4968];

%results of quantized network with 8x8 stride
%RMSE_test_doa =[0.8266    0.9311    1.6146];


%%
%qDetails = quantizationDetails(qNet)

%The QuantizedLayerNames field displays a list of quantized layers.

%qDetails.QuantizedLayerNames

%The QuantizedLearnables field contains additional details about the quantized network learnable parameters. In this example, the 2-D convolutional layers and fully connected layers have their weights scaled and cast to int8. The bias is scaled and remains in int32. The quantizationDetails function returns the values of the quantized learnables as stored integer values.

%qDetails.QuantizedLearnables


%%
%genero IP da utilizzare su FPGA

ConvThreadNumbers = [4, 9, 16, 25, 36, 49, 64]; %available values

Resources_DSP  = zeros (length(ConvThreadNumbers),1);
Resources_BRAM = zeros (length(ConvThreadNumbers),1);
Resources_LUT  = zeros (length(ConvThreadNumbers),1);
Latency        = zeros (length(ConvThreadNumbers),n_of_nets);

hPC_int8 = dlhdl.ProcessorConfig;

hdlsetuptoolpath('ToolName','Xilinx Vivado','ToolPath',...
 'D:\Program\Xilinx\Vivado\2020.2\bin\vivado.bat');

hPC_int8.SynthesisTool = 'Xilinx Vivado';


setModuleProperty(hPC_int8, 'conv', 'InputMemorySize', [256 128 1]);
setModuleProperty(hPC_int8, 'conv', 'OutputMemorySize', [64 64 1]);

setModuleProperty(hPC_int8, 'fc', 'InputMemorySize', 128);
setModuleProperty(hPC_int8, 'fc', 'OutputMemorySize', 128);

hPC_int8.TargetPlatform = 'Generic Deep Learning Processor';

hPC_int8.ProcessorDataType = 'int8';
hPC_int8.UseVendorLibrary = 'off'; %must be set to off if ProcessorDataType = 'int8'

for j = 1 : length(ConvThreadNumbers)
    setModuleProperty(hPC_int8, 'conv', 'ConvThreadNumber', ConvThreadNumbers(j));
    
    Resources = estimateResources(hPC_int8);
    
    Resources_DSP(j)  = Resources{"DL_Processor","DSP"};
    Resources_BRAM(j) = Resources{"DL_Processor","blockRAM"};
    Resources_LUT(j)  = Resources{"DL_Processor","LUT"};
        
    for i = 1 : n_of_nets
        Performance = estimatePerformance(hPC_int8, qNet_array{i});
        Latency(j,i)        = Performance{"Network","Latency(seconds)"}*1000;
    end
end
%%

figure;
for i = 1:n_of_nets
    figure;
    %subplot(n_of_nets,1,i);  % 3 righe, 1 colonna, subplot i
    hold on;
    idx_start = (i-1)*5 + 1;
    idx_end = i*5;
    yyaxis left;
    ylabel("Number of DSP")
    plot(Resources_DSP, 'DisplayName', 'Resources DSP');
    %plot(Resources_BRAM/max(Resources_BRAM), 'DisplayName', 'Resources_BRAM');
    %plot(Resources_LUT/max(Resources_LUT), 'DisplayName', 'Resources_LUT');
    yyaxis right;
    plot(Latency(:,i), 'DisplayName', 'Latency (ms)')

xticklabels(ConvThreadNumbers)

    xlabel("Convolutional Thread Number")
    ylabel("Latency (ms)")
    grid on;
    grid minor
    legend show;
    title('Quantized Pruned Network DLP IP');
    hold off;
end

%%
save (".\quantization_workspace.mat","Resources_DSP", "Resources_BRAM", "Resources_LUT", "Latency")


%%
j = 2; %conv set to 9
setModuleProperty(hPC_int8, 'conv', 'ConvThreadNumber', ConvThreadNumbers(j));

dlhdl.buildProcessor(hPC_int8,'ProjectFolder','db4raprocessor_conv9_prj',...
'ProcessorName','db4ra_processor','HDLCoderConfig',{'TargetLanguage','VHDL'});
