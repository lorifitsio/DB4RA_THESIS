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

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
%+++++++++++++++ SEZIONE SEGNALE UTILE ++++++++++++++++++++++++++++++++++
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% PARAMETRI DI CONFIGURAZIONE DEL SEGNALE UTILE
%imposto frequenza di start e stop della chirp di riferimento
f_start = 0;
f_stop_array  = [150 50 25 10 5]*1e6;
downsample_array = [6 18 36 90 180];

%f_stop_array  = [150 50]*1e6;
%downsample_array = [6 18];

oversampling_factor = (CollectorSampleRate./(2*downsample_array.*f_stop_array)) - 1; %fattore di oversampling, calcolato per ciascuna chirp. deve essere uguale per tutte

n_scenarios = length(f_stop_array); %NUMERO DI FREQUENZE UTILIZZATE PER GENERARE LE VARIE CHIRP

% genero un'array chirp che in 100 us passa linearmente da 0 alla frequenza target
x_array = zeros(n_scenarios,sample_length);
for i = 1 : n_scenarios
	x_array(i,:) = chirp(t,f_start,t(end),f_stop_array(i));
end

%VARIABILI PER GLI ANGOLI AZIMUTH CHE IL SEGNALE UTILE SPAZZA
% il segnale utile si muoverà per ogni iterazione dall'angolo azimuth_start
% all'angolo azimuth_stop, con un incremento pari ad azimuth_step

%sezione relativa alla porzione di dataset "without jammer" 
azimuth_start_woj = -60; % degrees
azimuth_stop_woj  = 60; % degrees
azimuth_step_woj = 2; % degrees. Qualsiasi valore reale maggiore di zero è valido per questo paramentro
elevation_angle_woj = 0; % tied constant
azimuth_angles_steps_woj = floor((azimuth_stop_woj-azimuth_start_woj)/azimuth_step_woj); %numero di step necessari per spazzare il range configurato

%sezione relativa alla porzione di dataset "with jammer" 
azimuth_start_wj = -60; % degrees
azimuth_stop_wj  = 60; % degrees
azimuth_step_wj = 4; % degrees. Qualsiasi valore reale maggiore di zero è valido per questo paramentro
elevation_angle = 0; % tied constant
azimuth_angles_steps_wj = floor((azimuth_stop_wj-azimuth_start_wj)/azimuth_step_wj); %numero di step necessari per spazzare il range configurato

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
%+++++++++++++++ FINE SEZIONE SEGNALE UTILE +++++++++++++++++++++++++++++
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
%+++++++++++++++ SEZIONE INTERFERENTE +++++++++++++++++++++++++++++++++++
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%JAMMER SECTION
%definisco le frequenze dei jammer sinusoidali
n_sin_jammers = 2;
sin_jammer_freq = ((1:n_sin_jammers)- 0.5)/n_sin_jammers; %[Hz] FREQUENZE JAMMER

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
azimuth_step_j = 2; % degrees. Qualsiasi valore reale maggiore di zero è valido per questo paramentro
elevation_angle_j = 0; % tied constant

t2j_distance = 20; %degrees: target to jammer distance

azimuth_angles_steps_j = floor((azimuth_stop_j-azimuth_start_j)/azimuth_step_j); %numero di step necessari per spazzare il range configurato
azimuth_angles_steps_reduced_j = floor(((azimuth_stop_j-t2j_distance)-(azimuth_start_j+t2j_distance))/azimuth_step_j); %numero di step possibili a causa della limitazione tra angolo di interferente e jammer

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
%+++++++++++++++ FINE SEZIONE INTERFERENTE ++++++++++++++++++++++++++++++
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%++++++++++++++ SEZIONE RUMORE ++++++++++++++++++++++++++++++++++++++++++
%inizializzo un rumore
snr = [30]; %dB array di rapporti snr
snr_length = length(snr);
P_chirp = (x_array(1,:)*(x_array(1,:))')/sample_length; %si utilizza come riferimento del rapporto segnale rumore, la potenza della chirp all'indice 1 del chirp array
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
n_dataset_samples_woj = snr_length*n_scenarios*(azimuth_angles_steps_woj+1);
n_dataset_samples_wj = snr_length*n_scenarios*(azimuth_angles_steps_wj+1)*(1 + azimuth_angles_steps_reduced_j*n_sin_jammers);
n_dataset_samples = n_dataset_samples_woj + n_dataset_samples_wj;

y_ds_cell = cell(n_dataset_samples, 5);

for i = 1 : n_dataset_samples
    y_ds_cell{i,1} = zeros(ceil((sample_length)/downsample_array(1)), n_antennas); 
end

for i = 1 : n_dataset_samples_woj
    y_ds_cell{i,3} = 0;
    y_ds_cell{i,4} = 0; 
    y_ds_cell{i,5} = 0; 
end

fir_taps = 128;
fir_coeffs = zeros (n_scenarios, fir_taps+1);

for i = 1 : n_scenarios
    %fir_coeffs(i,:) = fir1(fir_taps,f_stop_array(i)*2/CollectorSampleRate);%frequenza di taglio
                                                                            %impostata alla massima freqeunza della chirp 
    fir_coeffs(i,:) = fir1(fir_taps, 1/downsample_array(i));                %frequenza di taglio impostata a f_sampling/2, dove f_sampling 
                                                                            % è la frequenza di campionamento dopo il sottocampionamento
end

enable_jammer_jitter = 0;
y_ds_row = 1;

%ciclo la generazione per ciascuna delle dimensioni di snr_length
for n = 1 : snr_length

    noise_variance = P_chirp/(10^(snr(n)/10));%imposto la varianza del rumore sulla base dell'snr di riferimento
    % ciclo for per tutti gli scenari
    for m = 1 : n_scenarios
        azimuth_angle = azimuth_start_woj; %imposto l'angolo di partenza per la chirp in questione
        %spazzo tutto il range di angoli
        for i = 1 : azimuth_angles_steps_woj+1
            incidentAngle = [azimuth_angle;elevation_angle];
            noise = randn(sample_length, 1)*sqrt(noise_variance); %genero un vettore di rumore per ogni iterazione
    
            y = db4ra_collector((x_array(m,:))' + noise,incidentAngle); %calcolo l'uscita delle antenne della chirp che incide con l'angolo definito da incident angle
            
            %ciclo per il numero di antenne, poiché y è un array di 4 vettori, uno per ciascuna antenna
            for p = 1 : n_antennas
		    temp = downsample(conv(y(:,p),fir_coeffs(m,:),'same'), downsample_array(1)); %effettuo la decimazione
	            y_ds_cell{y_ds_row,1}(:,p) = temp; %salvo il l'uscita dell'antenna decimata nel dataset
            end
		    
            y_ds_cell{y_ds_row,2} = azimuth_angle;
            azimuth_angle = azimuth_angle + azimuth_step_woj; %incremento l'angolo
            y_ds_row = y_ds_row + 1;
        end
    end
end


%ciclo la generazione per ciascuna delle dimensioni di snr_length
for n = 1 : snr_length

    noise_variance = P_chirp/(10^(snr(n)/10));%imposto la varianza del rumore sulla base dell'snr di riferimento
    % ciclo for per tutti gli scenari
    for l = 1 : n_sin_jammers
        for m = 1 : n_scenarios
            azimuth_angle = azimuth_start_wj; %imposto l'angolo di partenza per la chirp in questione
            
            %spazzo tutto il range di angoli del segnale utile
            for i = 1 : azimuth_angles_steps_wj+1
                incidentAngle = [azimuth_angle;elevation_angle];
                azimuth_angle_j = azimuth_start_j;
                
                %spazzo tutto il range di angoli del segnale interferente
                for j = 1 : azimuth_angles_steps_j+1
                    
                    incidentAngle_j = [azimuth_angle_j;elevation_angle_j];
                    
                    if ((azimuth_angle_j < (azimuth_angle - 20)) || (azimuth_angle_j > (azimuth_angle + 20)))
                        noise = randn(sample_length, 1)*sqrt(noise_variance); %genero un vettore di rumore per ogni iterazione
                        noise_j = randn(sample_length, 1)*sqrt(noise_variance);
        
                        target = (x_array(m,:))' + noise;
                        jammer = sin(2*pi*sin_jammer_freq(l)*f_stop_array(1)*(t')+2*pi*rand*enable_jammer_jitter) + noise_j;
                        y = db4ra_collector([target,jammer],[incidentAngle,incidentAngle_j]); %calcolo l'uscita delle antenne della chirp che incide con l'angolo definito da incident angle
                        
                        %ciclo per il numero di antenne, poiché y è un array di 4 vettori, uno per ciascuna antenna
                        for p = 1 : n_antennas
			    temp = downsample(conv(y(:,p),fir_coeffs(m,:),'same'), downsample_array(1)); %effettuo la decimazione
                            y_ds_cell{y_ds_row,1}(:,p) = temp; %salvo il l'uscita dell'antenna decimata nel dataset
                        end

                        y_ds_cell{y_ds_row,2} = azimuth_angle;
                        y_ds_cell{y_ds_row,3} = azimuth_angle_j;
                        y_ds_cell{y_ds_row,4} = sin_jammer_freq(l)*f_stop_array(1);
                        y_ds_cell{y_ds_row,5} = 1; 
                        y_ds_row = y_ds_row + 1;
                    end
                    azimuth_angle_j = azimuth_angle_j + azimuth_step_j; %incremento l'angolo
                end
                azimuth_angle = azimuth_angle + azimuth_step_wj; %incremento l'angolo
            end
        end
    end
end

