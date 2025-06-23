%creo l'oggetto quantizzato per la rete di riferimento
quantObj = dlquantizer(prunedNetFineTrained,'ExecutionEnvironment','FPGA');

calResults = calibrate(quantObj,signals_Val);

%https://it.mathworks.com/help/deeplearning/ref/dlquantizer.quantize.html#mw_eebf44dc-5794-4132-a4e8-fac660e149ec
qNet = quantize (quantObj, 'ExponentScheme','Histogram');
%%
qDetails = quantizationDetails(qNet)

%The QuantizedLayerNames field displays a list of quantized layers.

qDetails.QuantizedLayerNames

%The QuantizedLearnables field contains additional details about the quantized network learnable parameters. In this example, the 2-D convolutional layers and fully connected layers have their weights scaled and cast to int8. The bias is scaled and remains in int32. The quantizationDetails function returns the values of the quantized learnables as stored integer values.

qDetails.QuantizedLearnables

%%
%test performance
YTest = predict(prunedNetTrained,signals_Test);
qYTest = predict(qNet,signals_Test);

%%
RMSE_test_doa=rmse(YTest(:,1),doa_Test)
qRMSE_test_doa=rmse(qYTest(:,1),doa_Test)


%%
hPC_int8 = dlhdl.ProcessorConfig;
hdlsetuptoolpath('ToolName','Xilinx Vivado','ToolPath',...
 'D:\Program\Xilinx\Vivado\2020.2\bin\vivado.bat');

hPC_int8.SynthesisTool = 'Xilinx Vivado';

setModuleProperty(hPC_int8, 'conv', 'ConvThreadNumber', 16);
setModuleProperty(hPC_int8, 'conv', 'InputMemorySize', [256 128 1]);
setModuleProperty(hPC_int8, 'conv', 'OutputMemorySize', [64 64 1]);

setModuleProperty(hPC_int8, 'fc', 'InputMemorySize', 128);
setModuleProperty(hPC_int8, 'fc', 'OutputMemorySize', 128);

hPC_int8.TargetPlatform = 'Generic Deep Learning Processor';

hPC_int8.ProcessorDataType = 'int8';
hPC_int8.UseVendorLibrary = 'off'; %must be set to off if ProcessorDataType = 'int8'

%%
dlhdl.buildProcessor(hPC_int8,'ProjectFolder','db4raprocessor_prj',...
'ProcessorName','db4ra_processor','HDLCoderConfig',{'TargetLanguage','VHDL'});

%%
hPC_single = dlhdl.ProcessorConfig;
hdlsetuptoolpath('ToolName','Xilinx Vivado','ToolPath',...
 'D:\Program\Xilinx\Vivado\2020.2\bin\vivado.bat');

hPC_single.SynthesisTool = 'Xilinx Vivado';

setModuleProperty(hPC_single, 'conv', 'ConvThreadNumber', 4);
setModuleProperty(hPC_single, 'conv', 'InputMemorySize', [256 128 1]);
setModuleProperty(hPC_single, 'conv', 'OutputMemorySize', [64 64 1]);

setModuleProperty(hPC_single, 'fc', 'InputMemorySize', 128);
setModuleProperty(hPC_single, 'fc', 'OutputMemorySize', 128);

hPC_single.TargetPlatform = 'Generic Deep Learning Processor';

hPC_single.ProcessorDataType = 'single';
hPC_single.UseVendorLibrary = 'on'; %must be set to off if ProcessorDataType = 'int8'

%%
dlhdl.buildProcessor(hPC_single,'ProjectFolder','db4raprocessor_single_prj',...
'ProcessorName','db4ra_processor_single','HDLCoderConfig',{'TargetLanguage','VHDL'});


%%
a = estimateResources(hPC_int8);
%%
estimateResources(hPC_single)
%%
b = estimatePerformance(hPC_single, qNet);
estimatePerformance(hPC_single, prunedNetFineTrained)
%%
estimatePerformance(hPC_int8, qNet)
estimatePerformance(hPC_int8, prunedNetFineTrained)

%%
estimatePerformance(hPC_single, net)
