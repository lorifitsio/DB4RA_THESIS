%%
% In questa porzione di codice configuro il Wideband Collector e genero i segnali di stimolo per le antenne

fc = 5.4e9; % 5.4 GHz carrier frequency
lambda = 0.027*2; %wavelength of maximum useful signal in band of interest (5.4 GHz + 150 MHz), related to inter antenna distance
n_antennas = 4;
% Element and ArrayAxis are kept default. Defaul element is Isoctropic,
% default axis is "y". Axis is important to correctly set azimuth angle.

myArray = phased.ULA("NumElements",n_antennas,"ElementSpacing",lambda/2);
ModulatedInputCtrl = true; %Set this property to true to indicate the input signal is demodulated at a carrier frequency.

CollectorCarrierFrequency = fc;
CollectorSampleRate = 2e9; %samplerate set equal to electra mx2 target throughput = 2 Gsamples
CollectorNumSubbands = 10000;

collector = phased.WidebandCollector("CarrierFrequency",CollectorCarrierFrequency, ...
    "ModulatedInput",ModulatedInputCtrl,"SampleRate",CollectorSampleRate, "Sensor",myArray, "NumSubbands",CollectorNumSubbands);

%genero l'asse dei tempi per la chirp, di durata 100 us
sample_width = 100e-6; %100 us length of chirp
sample_length = sample_width * CollectorSampleRate;
t = (0:sample_length-1)/CollectorSampleRate;

%imposto frequenza di start e stop della chirp di riferimento
f_start = 0;
f_stop_array  = [150 50 25 10 5]*1e6;
f_stop_array  = [150 50]*1e6;
downsample_array = [6 18 36 90 180];

n_scenarios = length(f_stop_array);
oversampling_factor = zeros(1,n_scenarios);

for i = 1 : n_scenarios
	oversampling_factor(i) = (2e9/downsample_array(i))/(2*f_stop_array(i)) - 1; 
end

% genero un'array chirp che in 100 us passa linearmente da 0 alla frequenza target
x_array = zeros(n_scenarios,sample_length);
for i = 1 : n_scenarios
	x_array(i,:) = chirp(t,f_start,t(end),f_stop_array(i));
end

sin_jammer_freq = [100 35]*1e6;
n_sin_jammers = length(sin_jammer_freq);

%%
% In questa porzione di codice genero un database con cinque
% segnali chirp diversi. Ciascuna chirp stimola l'array di antenne con 121
% angoli di incidenza diversa, da -60° a + 60° con step 1°.
% NB: questa configurazione è il default, il range di angoli e anche lo
% step sono diversamente configurabili
% L'antenna produce 4 vettori di segnali complessi per ciascun segnale con cui viene stimolata
% Il numero di chirp utilizzati è descritto dalla variabile "n_scenarios"
% Le varie chirp sono generate utilizzando una frequenza di campionamento
% di 2 GHz. Con tale frequenza di campionamento vengono mandate in input
% all'antenna. I segnali vengono successivamente filtrati e
% sottocampionati. Il sottocampionamento è tale per cui il rapporto tra la
% frequenza di campionamento e la banda bilatera della chirp è 11/9. 
% Esempio: chirp lineare da 0 Hz a 150 MHz, banda bilatera ad RF = 300 MHz (5,25 GHz - 5,55 GHz). Frequenza di
% campionamento dopo downsampling = 333 MHz
% Il downsampling viene effettuato utilizzando un fir 128 taps con
% coefficienti reali.


%definisco delle variabili per gli angoli
azimuth_start = -30; % degrees
azimuth_stop  = 30; % degrees
azimuth_angle = azimuth_start; % spanning between +60 and -60
elevation_angle = 0; % tied constant
azimuth_step = 2; % degrees. Qualsiasi valore reale maggiore di zero è valido per questo paramentro
incidentAngle = [azimuth_angle;elevation_angle];
incidentAngleSteps = floor((azimuth_stop-azimuth_start)/azimuth_step);

%inizializzo un rumore
snr = [30]; %dB array di rapporti snr
snr_length = length(snr);
P_chirp = (x_array(1,:)*(x_array(1,:))')/sample_length; %si utilizza come riferimento del rapporto segnale rumore, la potenza della chirp a 150 MHz

%JAMMER SECTION
%definisco delle variabili per gli angoli
azimuth_start_j = -50; % degrees
azimuth_stop_j  = 50; % degrees
azimuth_angle_j = azimuth_start; % spanning between +60 and -60
elevation_angle_j = 0; % tied constant
azimuth_step_j = 1; % degrees. Qualsiasi valore reale maggiore di zero è valido per questo paramentro
incidentAngle_j = [azimuth_angle_j;elevation_angle_j];
incidentAngleSteps_j = floor((azimuth_stop_j-azimuth_start_j)/azimuth_step_j);
incidentAngleSteps_reduced_j = floor(((azimuth_stop_j-20)-(azimuth_start_j+20))/azimuth_step_j);

% Inizializzo un cell array con due colonne: la prima con le uscite dell'antenna sottocampionate
% e la seconda con l'angolo di incidenza della chirp. La colonna ha lunghezza pari a snr_length*n_scenarios*(incidentAngleSteps+1)
% e contiene tutti i sample di ciascuna chirp, che incide sull'antenna con un numero di angoli pari a 
% "incidentAngleSteps+1". Se la variabile "snr" è un vettore, lo stessa popolazione del dataset
% verrà raccolta con i vari snr impostati

%inizializzo il cell array
%la lunghezza di ciascun segnale è pari a sample_length diviso il fattore
%di sottocampionamento della chirp a 150 MHz. tutti i segnali sono adeguati
%a questa trama, essendo il vettore più lungo
y_ds_cell = cell(snr_length*n_scenarios*(incidentAngleSteps+1)*incidentAngleSteps_reduced_j*n_sin_jammers, 4);
for i = 1 : snr_length*n_scenarios*(incidentAngleSteps+1)*incidentAngleSteps_reduced_j*n_sin_jammers
    y_ds_cell{i,1} = zeros(ceil((sample_length)/downsample_array(1)), ... 
        n_antennas); 
end


enable_jammer_jitter = 0;

%ciclo la generazione per ciascuna delle dimensioni di snr_length
for n = 1 : snr_length

    noise_variance = P_chirp/(10^(snr(n)/10));%imposto la varianza del rumore sulla base dell'snr di riferimento
    % ciclo for per tutti gli scenari
    for l = 1 : n_sin_jammers
        for m = 1 : n_scenarios
            azimuth_angle = azimuth_start; %imposto l'angolo di partenza per la chirp in questione
            
            %spazzo tutto il range di angoli
            for i = 1 : incidentAngleSteps+1
                incidentAngle = [azimuth_angle;elevation_angle];
                azimuth_angle_j = azimuth_start_j;
                j_reduced = 1;
                for j = 1 : incidentAngleSteps_j+1
                    
                    incidentAngle_j = [azimuth_angle_j;elevation_angle_j];
                    
                    if ((azimuth_angle_j < (azimuth_angle - 20)) || (azimuth_angle_j > (azimuth_angle + 20)))
                        noise = randn(sample_length, 1)*sqrt(noise_variance); %genero un vettore di rumore per ogni iterazione
                        noise_j = randn(sample_length, 1)*sqrt(noise_variance);
        
                        target = (x_array(m,:))' + noise;
                        jammer = sin(2*pi*sin_jammer_freq(l)*(t')+2*pi*rand*enable_jammer_jitter) + noise_j;
                        y = collector([target,jammer],[incidentAngle,incidentAngle_j]); %calcolo l'uscita delle antenne della chirp che incide con l'angolo definito da incident angle
                        
                        %ciclo per il numero di antenne, poiché y è un array di 4 vettori, uno per ciascuna antenna
                        for p = 1 : n_antennas
			                temp = decimate(y(:,p), downsample_array(m), 128, "fir"); %effettuo la decimazione
                            y_ds_row = (n-1)*n_sin_jammers*(n_scenarios)*(incidentAngleSteps+1)*incidentAngleSteps_reduced_j+...
                            (l-1)*(n_scenarios)*(incidentAngleSteps+1)*incidentAngleSteps_reduced_j + (m-1)*(incidentAngleSteps+1)*incidentAngleSteps_reduced_j+(i-1)*incidentAngleSteps_reduced_j+j_reduced;
	                        y_ds_cell{y_ds_row,1}(1:length(temp),p) = temp; %salvo il l'uscita dell'antenna decimata nel dataset
                        end
                        j_reduced = j_reduced+1;
                        y_ds_cell{y_ds_row,2} = azimuth_angle;
                        y_ds_cell{y_ds_row,3} = azimuth_angle_j;
                        y_ds_cell{y_ds_row,4} = sin_jammer_freq(l);
                    end
                    azimuth_angle_j = azimuth_angle_j + azimuth_step_j; %incremento l'angolo
                end
                azimuth_angle = azimuth_angle + azimuth_step; %incremento l'angolo
            end
        end
    end
end

