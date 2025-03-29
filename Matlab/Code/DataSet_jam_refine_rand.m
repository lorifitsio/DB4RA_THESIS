clc; clear; close all;

load ds_rand_raw.mat %y_ds da darabase gen
%y_ds_cell  = y_ds_cell(2001:7000,:);

%% Angles and jamming presence
n_samples = length(y_ds_cell);

doa=cell2mat(y_ds_cell(:,4)); %doa disturbo
jam=cell2mat(y_ds_cell(:,6)); % presenza o meno di jammimg


%%

signals = zeros (length(y_ds_cell{1,1}),8,1,n_samples);

for idx=1:n_samples

    temp_real=real(cell2mat(y_ds_cell(idx,1)));
    temp_imag=imag(cell2mat(y_ds_cell(idx,1)));

    signals(:,:,1,idx)=[temp_real temp_imag];
end

save("ds_rand.mat","signals", "doa", "jam", "-v7.3","-nocompression")
