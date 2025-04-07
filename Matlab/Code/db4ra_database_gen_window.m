%%
clc; clear; close all;


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
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
%+++++++++++++++ FINE SEZIONE INTERFERENTE ++++++++++++++++++++++++++++++
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%++++++++++++++ SEZIONE RUMORE ++++++++++++++++++++++++++++++++++++++++++
%inizializzo un rumore
snr = [20]; %dB array di rapporti snr
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
n_dataset_samples_woj = 18000;

n_chirp_angles = 1350;
n_jammer_angles_per_chirp = 20;
n_dataset_samples_wj = n_chirp_angles * n_jammer_angles_per_chirp;
n_dataset_samples = snr_length * (n_dataset_samples_woj + n_dataset_samples_wj);

y_ds_cell = cell(n_dataset_samples, 3);

signals = zeros (window_length/downsample_array(1),n_antennas*2,1,n_dataset_samples);
y_ds_temp = zeros (window_length/downsample_array(1),n_antennas);
doa = zeros(n_dataset_samples,1); %doa disturbo
jam = zeros(n_dataset_samples,1); % presenza o meno di jammimg

for i = 1 : n_dataset_samples_woj
    y_ds_cell{i,3} = 0;
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

%ciclo la generazione per ciascuna delle dimensioni di snr_length
for n = 1 : snr_length

    noise_variance = P_chirp/(10^(snr(n)/10));%imposto la varianza del rumore sulla base dell'snr di riferimento
    % ciclo for per tutti gli scenari

	azimuth_angle_vector = randi([azimuth_start_woj azimuth_stop_woj], 1, n_dataset_samples_woj);
	elevation_angle = elevation_angle_woj;
    for m = 1 : n_dataset_samples_woj
        azimuth_angle = azimuth_angle_vector(m); %imposto l'angolo di partenza per la chirp in questione
        %spazzo tutto il range di angoli
		incidentAngle = [azimuth_angle;elevation_angle];
		noise = randn(window_length, 1)*sqrt(noise_variance); %genero un vettore di rumore per ogni iterazione
		
		chirp_sel = randi([1 n_scenarios], 1, 1);
        window_sel = randi([1 window_length], 1, 1);
			
		y = db4ra_collector((x_array(chirp_sel,window_sel:window_sel+window_length-1))' + noise,incidentAngle); %calcolo l'uscita delle antenne della chirp che incide con l'angolo definito da incident angle
		
		%ciclo per il numero di antenne, poiché y è un array di 4 vettori, uno per ciascuna antenna
		for p = 1 : n_antennas
			temp = downsample(conv(y(:,p),fir_coeffs(chirp_sel,:),'same'), downsample_array(1)); %effettuo la decimazione
			y_ds_temp(:,p) = temp; %salvo il l'uscita dell'antenna decimata nel dataset
		end
		
		signals(:,:,1,y_ds_row)=[real(y_ds_temp) imag(y_ds_temp)];

		y_ds_cell{y_ds_row,1} = azimuth_angle;
		y_ds_cell{y_ds_row,2} = f_stop_array(chirp_sel);
		y_ds_row = y_ds_row + 1;
    end
end

%%
%ciclo la generazione per ciascuna delle dimensioni di snr_length
for n = 1 : snr_length

    noise_variance = P_chirp/(10^(snr(n)/10));%imposto la varianza del rumore sulla base dell'snr di riferimento
    % ciclo for per tutti gli angoli del segnale utile
	azimuth_angle_vector = randi([azimuth_start_wj azimuth_stop_wj], 1, n_chirp_angles);
	
    for m = 1 : n_chirp_angles
            
		azimuth_angle = azimuth_angle_vector(m); %imposto l'angolo di partenza per la chirp in questione

		azimuth_angle_vector_j = randi([azimuth_start_j+t2j_distance azimuth_stop_j-t2j_distance-1], 1, n_jammer_angles_per_chirp);

		sel = azimuth_angle_vector(m) > azimuth_angle_vector_j;
		azimuth_angle_vector_j(sel) = azimuth_angle_vector_j(sel) - t2j_distance;
		
		sel = azimuth_angle_vector(m) <= azimuth_angle_vector_j;
		azimuth_angle_vector_j(sel) = azimuth_angle_vector_j(sel) + t2j_distance+1;

        chirp_sel = randi([1 n_scenarios], 1, 1);
        if (chirp_sel == 5) 
	        jammer_freq = randi([2 f_stop_array(chirp_sel)/(1e6)], 1, 1) * (1e6);
        else			
	        jammer_freq = randi([f_stop_array(chirp_sel+1)/(1e6) f_stop_array(chirp_sel)/(1e6)], 1, 1) * (1e6);
        end
		%spazzo tutto il range di angoli del segnale utile
		for i = 1 : n_jammer_angles_per_chirp
			incidentAngle = [azimuth_angle;elevation_angle];
			azimuth_angle_j = azimuth_angle_vector_j(i);
			
		    %spazzo tutto il range di angoli del segnale interferente
				
			incidentAngle_j = [azimuth_angle_j;elevation_angle_j];
		    noise = randn(window_length, 1)*sqrt(noise_variance); %genero un vettore di rumore per ogni iterazione
			
            chirp_sel = randi([1 n_scenarios], 1, 1);
            window_sel = randi([1 window_length], 1, 1);
		            
            noise_j = randn(window_length, 1)*sqrt(noise_variance);
											
			target = (x_array(chirp_sel,window_sel:window_sel+window_length-1))' + noise;
        												  
			jammer = sin(2*pi*jammer_freq*(t(1:window_length)')+2*pi*rand*enable_jammer_jitter) + noise_j;
			y = db4ra_collector([target,jammer],[incidentAngle,incidentAngle_j]); %calcolo l'uscita delle antenne della chirp che incide con l'angolo definito da incident angle
			
			%ciclo per il numero di antenne, poiché y è un array di 4 vettori, uno per ciascuna antenna
			for p = 1 : n_antennas
				temp = downsample(conv(y(:,p),fir_coeffs(chirp_sel,:),'same'), downsample_array(1)); %effettuo la decimazione
				y_ds_temp(:,p) = temp; %salvo il l'uscita dell'antenna decimata nel dataset
			end

            signals(:,:,1,y_ds_row)=[real(y_ds_temp) imag(y_ds_temp)];

            y_ds_cell{y_ds_row,1} = azimuth_angle;
		    y_ds_cell{y_ds_row,2} = f_stop_array(chirp_sel);
        	doa(y_ds_row) = azimuth_angle_j;
			y_ds_cell{y_ds_row,3} = jammer_freq;
			jam(y_ds_row) = 1; 
			y_ds_row = y_ds_row + 1;
		end
    end
end

%%
err_count = 0;
for i = (snr_length * n_dataset_samples_woj + 1) : (snr_length * n_dataset_samples_wj)
    distance = abs(doa(i) - y_ds_cell{i,1});
    if (distance <= 20)
        err_count = err_count +1;
    end
end

if (err_count > 0)
    fprintf(1,'There was an error!');
end
 
err_count

    % try
    %    %Error-maker
    % catch e %e is an MException struct
    %     fprintf(1,'The identifier was:\n%s',e.identifier);
    %     fprintf(1,'There was an error! The message was:\n%s',e.message);
    %     % more error handling...
    % end

%%
save("ds_window.mat","y_ds_cell", "doa", "jam", "signals","-v7.3","-nocompression")

