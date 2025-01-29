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
azimuth_start = -60; % degrees
azimuth_stop  = 60; % degrees
azimuth_angle = azimuth_start; % spanning between +60 and -60
elevation_angle = 0; % tied constant
azimuth_step = 1; % degrees. Qualsiasi valore reale maggiore di zero è valido per questo paramentro
incidentAngle = [azimuth_angle;elevation_angle];
incidentAngleSteps = floor((azimuth_stop-azimuth_start)/azimuth_step);

%inizializzo un rumore
snr = [30 50]; %dB array di rapporti snr
snr_length = length(snr);
P_chirp = (x_array(1,:)*(x_array(1,:))')/sample_length; %si utilizza come riferimento del rapporto segnale rumore, la potenza della chirp a 150 MHz

% Inizializzo l'array 4D dove salvo le uscite delle antenne campionate a 2
% GHz. 
% la riga sottostante è stata commentata. non viene più generato il
% vettore 4d contenente tutto il database delle chirp campionate a
% 2 GHz. Utilizzo un vettore temporaneo per salvare l'uscita del
% wideband collector. In questo modo accelero l'algoritmo.
%y = zeros(n_scenarios, sample_length,n_antennas,incidentAngleSteps+1);

% Inizializzo un cell array con due colonne: la prima con le uscite dell'antenna sottocampionate
% e la seconda con l'angolo di incidenza della chirp. La colonna ha lunghezza pari a snr_length*n_scenarios*(incidentAngleSteps+1)
% e contiene tutti i sample di ciascuna chirp, che incide sull'antenna con un numero di angoli pari a 
% "incidentAngleSteps+1". Se la variabile "snr" è un vettore, lo stessa popolazione del dataset
% verrà raccolta con i vari snr impostati

%inizializzo il cell array
%la lunghezza di ciascun segnale è pari a sample_length diviso il fattore
%di sottocampionamento della chirp a 150 MHz. tutti i segnali sono adeguati
%a questa trama, essendo il vettore più lungo
y_ds_cell = cell(snr_length*n_scenarios*(incidentAngleSteps+1), 2);
for i = 1 : snr_length*n_scenarios*(incidentAngleSteps+1)
    y_ds_cell{i,1} = zeros(ceil((sample_length)/downsample_array(1)), ... 
        n_antennas); 
end

%ciclo la generazione per ciascuna delle dimensioni di snr_length
for n = 1 : snr_length

    noise_variance = P_chirp/(10^(snr(n)/10));%imposto la varianza del rumore sulla base dell'snr di riferimento
    % ciclo for per tutti gli scenari
    for m = 1 : n_scenarios
        azimuth_angle = azimuth_start; %imposto l'angolo di partenza per la chirp in questione
        
        %spazzo tutto il range di angoli
        for i = 1 : incidentAngleSteps+1
            incidentAngle = [azimuth_angle;elevation_angle];
            noise = randn(sample_length, 1)*sqrt(noise_variance); %genero un vettore di rumore per ogni iterazione
    
            y = collector((x_array(m,:))' + noise,incidentAngle); %calcolo l'uscita delle antenne della chirp che incide con l'angolo definito da incident angle
            
            %ciclo per il numero di antenne, poiché y è un array di 4 vettori, uno per ciascuna antenna
            for p = 1 : n_antennas
			    temp = decimate(y(:,p), downsample_array(m), 128, "fir"); %effettuo la decimazione
                y_ds_row = (n-1)*(n_scenarios)*(incidentAngleSteps+1)+(m-1)*(incidentAngleSteps+1)+i;
	            y_ds_cell{y_ds_row,1}(1:length(temp),p) = temp; %salvo il l'uscita dell'antenna decimata nel dataset
            end
		    
            y_ds_cell{y_ds_row,2} = azimuth_angle;
            azimuth_angle = azimuth_angle + azimuth_step; %incremento l'angolo
        end
    end
end

