%%
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
%ESEMPIO DI USCITA DEL WIDEBAND COLLECTOR
%Le antenne ricevono una chirp lineare da 0 MHz a f_stop MHz, proveniente da
%un angolo azimuth arbitrario e un angolo elevation pari a 0°

%definisco delle variabili per gli angoli
azimuth_angle = 60; % spanning between +60 and -60
elevation_angle = 0; % tied constant
incidentAngle = [azimuth_angle;elevation_angle];

%imposto quale dei cinque scenari utilizzare in questo esempio
s = 1;

%inizializzo un rumore
snr = 30; %dB
P_chirp = (x_array(1,:)*(x_array(1,:))')/sample_length;
noise_variance = P_chirp/(10^(snr/10));

%inizializzo il vettore sottocampionato
y_ds = zeros(ceil((sample_length)/downsample_array(s)), n_antennas); 

% genero l'uscita dell'antenna y e la sua versione sotto campionata
noise = randn(sample_length, 1)*sqrt(noise_variance);
y = collector((x_array(s,:))' + noise,incidentAngle); %calcolo l'uscita delle antenne della chirp che incide con l'angolo definito da incident angle
for p = 1 : n_antennas
    y_ds(:,p) = decimate(y(:,p), downsample_array(s), 128, "fir"); 
end

%%
%plot del segnale x, y e y sottocampionata

f_2GHz = (1:sample_length)*1/sample_width;
f_ds   = f_2GHz(1:length(y_ds));

figure(1)
plot(f_2GHz,20*log10(abs(fft(x_array(s,:) + noise'))/max(abs(fft(x_array(s,:) + noise')))))
xlabel('f [Hz]')
ylabel('x [dB]')

figure(2)
spectrogram(x_array(s,:) + noise',[],[],[],"centered")
      
figure(3)
plot(f_2GHz,20*log10(abs(fft(y(:,1)))/max(abs(fft(y(:,1))))))
xlabel('f [Hz]')
ylabel('y [dB]')

figure(4)
spectrogram(y(:,1),[],[],[],"centered")

figure(5)
plot(f_ds,20*log10(abs(fft(y_ds(:,1))/max(abs(fft(y_ds(:,1)))))))
xlabel('f [Hz]')
ylabel('y ds [dB]')

figure(6)
spectrogram(y_ds(:,1),[],[],[],"centered")

%%
% In questa porzione di codice genero un database con cinque
% segnali chirp diversi. Ciascuna chirp stimola l'array di antenne con 121
% angoli di incidenza diversa, da -60° a + 60° con step 1°.
% L'antenna produce 4 vettori di segnali complessi per ciascun segnale con cui viene stimolata
% Il numero di chirp utilizzati è descritto dalla variabile "n_scenarios"
% Le varie chirp sono generate utilizzando una frequenza di campionamento
% di 2 GHz. Con tale frequenza di campionamento vengono mandate in input
% all'antenna. I segnali vengono successivamente filtrati e
% sottocampionati. Il sottocampionamento è tale per cui il rapporto tra la
% frequenza di campionamento e la banda bilatera della chirp è 4/3. 
% Esempio: chirp lineare da 0 Hz a 150 MHz, banda bilatera ad RF = 300 MHz (5,25 GHz - 5,55 GHz). Frequenza di
% campionamento dopo downsampling = 400 MHz
% Il downsampling viene effettuato utilizzando un fir 128 taps con
% coefficienti reali.


%definisco delle variabili per gli angoli
azimuth_start = -60; % degrees
azimuth_stop  = 60; % degrees
azimuth_angle = azimuth_start; % spanning between +60 and -60
elevation_angle = 0; % tied constant
azimuth_step = 1; % degrees
incidentAngle = [azimuth_angle;elevation_angle];
incidentAngleSteps = (azimuth_stop-azimuth_start)/azimuth_step;

% Inizializzo l'array 4D dove salvo le uscite delle antenne campionate a 2
% GHz. 
% la riga sottostante è stata commentata. non viene più generato il
% vettore 4d contenente tutto il database delle chirp campionate a
% 2 GHz. Utilizzo un vettore temporaneo per salvare l'uscita del
% widdeband collector. In questo modo accelero l'algoritmo.
%y = zeros(n_scenarios, sample_length,n_antennas,incidentAngleSteps+1);

% Inizializzo un cell array con dimensioni 1 x n_scenarios. In ciascuna cella salvo le 
% uscite delle antenne sottocampionate. è necessario utilizzare un cell array perché utilizzando 
% fattori di sottocampionamento diversi, gli array avranno lunghezza diversa 
y_ds_cell = cell(n_scenarios*(incidentAngleSteps+1), 2);
for i = 1 : n_scenarios*(incidentAngleSteps+1)
    y_ds_cell{i,1} = zeros(ceil((sample_length)/downsample_array(1)), ... 
        n_antennas); 
end


%inizializzo un rumore
snr = 30; %dB
P_chirp = (x_array(1,:)*(x_array(1,:))')/sample_length;
noise_variance = P_chirp/(10^(snr/10));

% ciclo for per la generazione del database
for m = 1 : n_scenarios
    azimuth_angle = azimuth_start;
    for i = 1 : incidentAngleSteps+1
        incidentAngle = [azimuth_angle;elevation_angle];
        noise = randn(sample_length, 1)*sqrt(noise_variance);

        % la riga sottostante è stata commentata. non viene più generato il
        % vettore 4d contenente tutto il database delle chirp campionate a
        % 2 GHz. Utilizzo un vettore temporaneo per salvare l'uscita del
        % widdeband collector. In questo modo accelero l'algoritmo.
        %y(m,:,:,i) = collector((x_array(m,:))' + noise,incidentAngle); %calcolo l'uscita delle antenne della chirp che incide con l'angolo definito da incident angle
        y = collector((x_array(m,:))' + noise,incidentAngle); %calcolo l'uscita delle antenne della chirp che incide con l'angolo definito da incident angle
        for p = 1 : n_antennas
			temp = decimate(y(:,p), downsample_array(m), 128, "fir");
	        y_ds_cell{(m-1)*(incidentAngleSteps+1)+i,1}(1:length(temp),p) = temp; 
        end
		y_ds_cell{(m-1)*(incidentAngleSteps+1)+i,2} = azimuth_angle;
        azimuth_angle = azimuth_angle + azimuth_step;
    end
end

