clc; clear; close all;

load ds_raw_1.5.2.5.mat %y_ds da darabase gen

%% Angles and jamming presence

n_samples = length(y_ds_cell);

doa=cell2mat(y_ds_cell(:,3)); %doa disturbo
jam=cell2mat(y_ds_cell(:,5)); % presenza o meno di jammimg


%%
signals = zeros (length(y_ds_cell{1,1}),8,1,n_samples);

for idx=1:n_samples

    temp_real=real(cell2mat(y_ds_cell(idx,1)));
    temp_imag=imag(cell2mat(y_ds_cell(idx,1)));

    signals(:,:,1,idx)=[temp_real temp_imag];
end

save("ds_1.5.2.5.mat","signals", "doa", "jam", "-v7.3","-nocompression")

