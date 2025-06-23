%%
clc; clear;close  all;

%%
% In questa porzione di codice configuro il Wideband Collector e genero i segnali di stimolo per le antenne

fc = 5.4e9; % 5.4 GHz carrier frequency
lambda = 0.027*2; %wavelength of maximum useful signal in band of interest (5.4 GHz + 150 MHz), related to inter antenna distance
n_antennas = 4;
% Element and ArrayAxis are kept default. Defaul element is Isoctropic,
% default axis is "y". Axis is important to correctly set azimuth angle.

db4ra_ula = phased.ULA("NumElements",n_antennas,"ElementSpacing",lambda/2);
ModulatedInputCtrl = true; %Set this property to true to indicate the input signal is demodulated at a carrier frequency.

CollectorCarrierFrequency = fc;
CollectorSampleRate = 2e9; %samplerate set equal to electra mx2 target throughput = 2 Gsamples
CollectorNumSubbands = 10000;

db4ra_collector = phased.WidebandCollector("CarrierFrequency",CollectorCarrierFrequency, ...
    "ModulatedInput",ModulatedInputCtrl,"SampleRate",CollectorSampleRate, "Sensor",db4ra_ula, "NumSubbands",CollectorNumSubbands);

%genero l'asse dei tempi per la chirp, di durata 100 us
sample_width = 100e-6; %100 us LENGTH OF CHIRP SIGNAL

sample_length = sample_width * CollectorSampleRate;
t = (0:sample_length-1)/CollectorSampleRate;

window_length = 24576; %la lunghezza impostata a 24576 consente di aver una finestra pari a 4096 (2^12) dopo il sottocampionamento
window_width = window_length / CollectorSampleRate; %12.23 us LENGTH OF CHIRP WINDOW with window_length = 24576

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
%+++++++++++++++ SEZIONE SEGNALE UTILE ++++++++++++++++++++++++++++++++++
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% PARAMETRI DI CONFIGURAZIONE DEL SEGNALE UTILE
%imposto frequenza di start e stop della chirp di riferimento
f_start = 0;
f_stop_array  = [150 50 25 10 5]*1e6;
downsample_array = [6 18 36 90 180];

n_scenarios = length(f_stop_array); %NUMERO DI FREQUENZE UTILIZZATE PER GENERARE LE VARIE CHIRP

% genero un'array chirp che in 100 us passa linearmente da 0 alla frequenza target
% appendo window_length campioni nulli alla chirp, sia prima che dopo, per poter emulare lo scenario di chirp intercettata
% parzialmente  
x_array = zeros(n_scenarios,sample_length+2*(window_length-1));
for i = 1 : n_scenarios
	x_array(i,:) = [zeros(1,window_length-1) chirp(t,f_start,t(end),f_stop_array(i)) zeros(1,window_length-1)];
end

%VARIABILI PER GLI ANGOLI AZIMUTH CHE IL SEGNALE UTILE SPAZZA
% il segnale utile si muoverà per ogni iterazione dall'angolo azimuth_start
% all'angolo azimuth_stop, con un incremento pari ad azimuth_step

%sezione relativa alla porzione di dataset "without jammer" 
azimuth_start_woj = -60; % degrees
azimuth_stop_woj  = 60; % degrees
elevation_angle_woj = 0; % tied constant

%sezione relativa alla porzione di dataset "with jammer" 
azimuth_start_wj = -60; % degrees
azimuth_stop_wj  = 60; % degrees
elevation_angle_wj = 0; % tied constant

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
%+++++++++++++++ FINE SEZIONE SEGNALE UTILE +++++++++++++++++++++++++++++
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
%+++++++++++++++ SEZIONE INTERFERENTE +++++++++++++++++++++++++++++++++++
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%JAMMER SECTION
%definisco le frequenze dei jammer sinusoidali
n_sin_jammers = 2;

%VARIABILI PER GLI ANGOLI AZIMUTH CHE IL SEGNALE INTERFERENTE SPAZZA
% il segnale interferente si muoverà per ogni iterazione dall'angolo
% azimuth_start_j all'angolo azimuth_stop_j, con un incremento pari ad
% azimuth_step_j. Non tutto il range però è disponibile: l'interferente si
% deve trovare ad almeno +-20° di distanza dal segnale utile. La variabile
% azimuth_angles_steps_reduced_j conta il numero effettivo di combinazioni
% dell'angolo dell'interfente per ciascun angolo del segnale utile, sulla
% base dei parametri di configurazione
azimuth_start_j = -80; % degrees
azimuth_stop_j  = 80; % degrees
elevation_angle_j = 0; % tied constant

t2j_distance = 20; %degrees: target to jammer distance

%definisco il jammer swept cw
% il jammer è una chirp che va da 0 a 300 MHz con un incremento di 200
% MHz/ms.
% Genero una chirp di durata 1,5 ms
sample_width_sweptcw = 1.5e-3; %1.5 ms LENGTH OF CHIRP SIGNAL

sample_length_sweptcw = sample_width_sweptcw * CollectorSampleRate;
t_sweptcw = (0:sample_length_sweptcw-1)/CollectorSampleRate;
sweptcw = chirp(t_sweptcw,0,t_sweptcw(end),300e6); 

sweptcw = [sweptcw(end - (window_length/2-1): end) sweptcw];
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
%+++++++++++++++ FINE SEZIONE INTERFERENTE ++++++++++++++++++++++++++++++
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%++++++++++++++ SEZIONE RUMORE ++++++++++++++++++++++++++++++++++++++++++
%inizializzo un rumore
snr = [30]; %dB array di rapporti snr
snr_length = length(snr);
P_chirp = (x_array(1,window_length:end-window_length+1)*(x_array(1,window_length:end-window_length+1))')/sample_length; %si utilizza come riferimento del rapporto segnale rumore, la potenza della chirp all'indice 1 del chirp array
%++++++++++++++ FINE SEZIONE RUMORE +++++++++++++++++++++++++++++++++++++

%%
%CODICE VERSIONE 2.0: IL DATASET NON E' PIU' COMPOSTO DA SEGNALI CON
%SOTTOCAMPIONAMENTO DIVERSO. IL SOTTOCAMPIONAMENTO E' UNICO, MA I CANALI
%VENGONO OPPORTUNAMENTE FILTRATI SULLA BASE DELLA FREQUENZA DELLA CHIRP

% In questa porzione di codice genero il dataset

%inizializzo il cell array
%la lunghezza di ciascun segnale è pari a sample_length diviso il fattore
%di sottocampionamento della chirp a 150 MHz. tutti i segnali sono adeguati
%a questa trama, essendo il vettore più lungo
n_dataset_samples_woj = 20000; %prima era 25000

n_chirp_angles = 900; %prima era 1300
n_jammer_angles_per_chirp = 25;
n_dataset_samples_wj = n_chirp_angles * n_jammer_angles_per_chirp;
n_dataset_samples = snr_length * (n_dataset_samples_woj + 4*n_dataset_samples_wj);

y_ds_cell = zeros(n_dataset_samples, 3);

signals = zeros (window_length/downsample_array(1),n_antennas*2,1,n_dataset_samples);
y_ds_temp = zeros (window_length/downsample_array(1),n_antennas);
doa = zeros(n_dataset_samples,1); %doa disturbo
jam = zeros(n_dataset_samples,1); % presenza o meno di jammimg

for i = 1 : n_dataset_samples_woj
    y_ds_cell(i,3) = 0;
    jam(i) = -1; % -1 if no jammer is present
end

fir_taps = 128;
fir_coeffs = zeros (n_scenarios, fir_taps+1);

for i = 1 : n_scenarios
    fir_coeffs(i,:) = fir1(fir_taps,f_stop_array(i)*2/CollectorSampleRate); %frequenza di taglio
                                                                            %impostata alla massima freqeunza della chirp 
    %fir_coeffs(i,:) = fir1(fir_taps, 1/downsample_array(i));                %frequenza di taglio impostata a f_sampling/2, dove f_sampling 
                                                                            % è la frequenza di campionamento dopo il sottocampionamento
end

enable_jammer_jitter = 0;
y_ds_row = 1;

%%ciclo la generazione per ciascuna delle dimensioni di snr_length
%for n = 1 : snr_length
%
%    noise_variance = P_chirp/(10^(snr(n)/10));%imposto la varianza del rumore sulla base dell'snr di riferimento
%    % ciclo for per tutti gli scenari
%
%	azimuth_angle_vector = randi([azimuth_start_woj azimuth_stop_woj], 1, n_dataset_samples_woj);
%	elevation_angle = elevation_angle_woj;
%    for m = 1 : n_dataset_samples_woj
%        azimuth_angle = azimuth_angle_vector(m); %imposto l'angolo di partenza per la chirp in questione
%        %spazzo tutto il range di angoli
%		incidentAngle = [azimuth_angle;elevation_angle];
%		noise = randn(window_length, 1)*sqrt(noise_variance); %genero un vettore di rumore per ogni iterazione
%		
%		chirp_sel = randi([1 n_scenarios], 1, 1);
%        window_sel = randi([1 window_length], 1, 1);
%			
%		y = db4ra_collector((x_array(chirp_sel,window_sel:window_sel+window_length-1))' + noise,incidentAngle); %calcolo l'uscita delle antenne della chirp che incide con l'angolo definito da incident angle
%		
%		%ciclo per il numero di antenne, poiché y è un array di 4 vettori, uno per ciascuna antenna
%		for p = 1 : n_antennas
%			temp = downsample(conv(y(:,p),fir_coeffs(chirp_sel,:),'same'), downsample_array(1)); %effettuo la decimazione
%			y_ds_temp(:,p) = temp; %salvo il l'uscita dell'antenna decimata nel dataset
%		end
%		
%		signals(:,:,1,y_ds_row)=[real(y_ds_temp) imag(y_ds_temp)];
%
%		y_ds_cell{y_ds_row,1} = azimuth_angle;
%		y_ds_cell{y_ds_row,2} = f_stop_array(chirp_sel);
%		y_ds_row = y_ds_row + 1;
%    end
%end

n = 1;
noise_variance = P_chirp/(10^(snr(n)/10));%imposto la varianza del rumore sulla base dell'snr di riferimento
azimuth_angle_vector = randi([azimuth_start_woj azimuth_stop_woj], 1, n_dataset_samples_woj);
elevation_angle = elevation_angle_woj;

[signals(:,:,:,1:n_dataset_samples_woj), y_ds_cell(1:n_dataset_samples_woj,:)] = db4ra_database_functions(db4ra_collector, f_stop_array,n_dataset_samples_woj, azimuth_angle_vector, elevation_angle, noise_variance, window_length, n_scenarios, sample_length, x_array, n_antennas, fir_coeffs, downsample_array);


%%

noise_variance = P_chirp/(10^(snr(n)/10));%imposto la varianza del rumore sulla base dell'snr di riferimento
% ciclo for per tutti gli angoli del segnale utile
azimuth_angle_vector = randi([azimuth_start_wj azimuth_stop_wj], 1, n_chirp_angles);
remove_signal = false;
intervallo = n_dataset_samples_woj+1:n_dataset_samples_woj+n_dataset_samples_wj;
[signals(:,:,:,intervallo), y_ds_cell(intervallo,:), doa(intervallo), jam(intervallo)] = db4ra_database_sin_jammer (remove_signal,enable_jammer_jitter,t,db4ra_collector,  azimuth_start_j, t2j_distance, azimuth_stop_j, f_stop_array,n_chirp_angles, n_jammer_angles_per_chirp, azimuth_angle_vector, elevation_angle, elevation_angle_j,noise_variance, window_length, n_scenarios, sample_length, x_array, n_antennas, fir_coeffs, downsample_array);

remove_signal = true;
	
intervallo = intervallo + n_dataset_samples_wj;
[signals(:,:,:,intervallo), y_ds_cell(intervallo,:), doa(intervallo), jam(intervallo)] = db4ra_database_sin_jammer (remove_signal,enable_jammer_jitter,t,db4ra_collector,  azimuth_start_j, t2j_distance, azimuth_stop_j, f_stop_array,n_chirp_angles, n_jammer_angles_per_chirp, azimuth_angle_vector, elevation_angle, elevation_angle_j,noise_variance, window_length, n_scenarios, sample_length, x_array, n_antennas, fir_coeffs, downsample_array);


remove_signal = false;
sweptcw_window_Sel = (ceil(f_stop_array/100)*1.05);
noise_variance = P_chirp/(10^(snr(n)/10));%imposto la varianza del rumore sulla base dell'snr di riferimento
% ciclo for per tutti gli angoli del segnale utile
azimuth_angle_vector = randi([azimuth_start_wj azimuth_stop_wj], 1, n_chirp_angles);
	
	
intervallo = intervallo + n_dataset_samples_wj;
%%
[signals(:,:,:,intervallo), y_ds_cell(intervallo,:), doa(intervallo), jam(intervallo)]  = db4ra_database_sweptcw_jammer (sweptcw,remove_signal, sweptcw_window_Sel, db4ra_collector, azimuth_start_j, t2j_distance, azimuth_stop_j, f_stop_array,n_chirp_angles, n_jammer_angles_per_chirp, azimuth_angle_vector, elevation_angle, elevation_angle_j, noise_variance, window_length, n_scenarios, sample_length, x_array, n_antennas, fir_coeffs, downsample_array);


azimuth_angle_vector = randi([azimuth_start_wj azimuth_stop_wj], 1, n_chirp_angles);
remove_signal = true;	
	
intervallo = intervallo + n_dataset_samples_wj;
[signals(:,:,:,intervallo), y_ds_cell(intervallo,:), doa(intervallo), jam(intervallo)]  = db4ra_database_sweptcw_jammer (sweptcw,remove_signal, sweptcw_window_Sel, db4ra_collector, azimuth_start_j, t2j_distance, azimuth_stop_j, f_stop_array,n_chirp_angles, n_jammer_angles_per_chirp, azimuth_angle_vector, elevation_angle, elevation_angle_j, noise_variance, window_length, n_scenarios, sample_length, x_array, n_antennas, fir_coeffs, downsample_array);


% %%
% index = index + 1;
% index_full = n_dataset_samples_woj +2*n_dataset_samples_wj + index*30;
% 
% figure(1);
% subplot(2,1,1)
% plot(signals(:,1,1,index_full))
% subplot(2,1,2)
% plot(10*log10(abs(fft(signals(:,1,1,index_full)))))
% (y_ds_cell(index_full,2))
% 
% (y_ds_cell(index_full,3))
%%

doa = doa/100*8; %normalizzo rispetto all'angolo 100° e moltiplico per 8, in modo che 100° sia rappresentato da +8 e 100 da -8

save("ds_window_swept.mat","y_ds_cell", "doa", "jam", "signals","-v7.3","-nocompression")

