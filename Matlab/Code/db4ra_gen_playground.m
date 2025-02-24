%%
%Questo codice permette di "giocare" con il wideband collector e visualizzare le uscite delle antenne
%a seguito di vari scenari arbitrari di configurazione

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
%Le antenne ricevono una chirp lineare da f_start MHz a f_stop MHz, proveniente da
%un angolo azimuth arbitrario e un angolo elevation pari a 0°

%definisco delle variabili per gli angoli
azimuth_angle = 60; % spanning between +60 and -60
elevation_angle = 0; % tied constant
incidentAngle = [azimuth_angle;elevation_angle];

%imposto quale dei cinque scenari utilizzare in questo esempio: s = 1 --> 150 MHz, s = 5 --> 5 MHz
s = 1;

%inizializzo un rumore
%uso come potenza di segnale utile la potenza della chirp a 150 MHz
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
%ESEMPIO DI USCITA DEL WIDEBAND COLLECTOR: INTRODUCO JAMMER SINUSOIDALE
%Le antenne ricevono una chirp lineare da f_start MHz a f_stop MHz, proveniente da
%un angolo azimuth arbitrario e un angolo elevation pari a 0°

%definisco delle variabili per gli angoli
azimuth_angle = 60; % spanning between +60 and -60
elevation_angle = 0; % tied constant
incidentAngle = [azimuth_angle;elevation_angle];

%JAMMER SECTION
%definisco delle variabili per gli angoli
azimuth_angle_j = -43; % degrees
elevation_angle_j = 0; % tied constant
incidentAngle_j = [azimuth_angle_j;elevation_angle_j];
sin_jammer_freq = [100]*1e6;
enable_jammer_jitter = 0;

%imposto quale dei cinque scenari utilizzare in questo esempio: s = 1 --> 150 MHz, s = 5 --> 5 MHz
s = 1;
j = 1;

%inizializzo un rumore
%uso come potenza di segnale utile la potenza della chirp a 150 MHz
snr = 30; %dB
P_chirp = (x_array(1,:)*(x_array(1,:))')/sample_length;
noise_variance = P_chirp/(10^(snr/10));

%inizializzo il vettore sottocampionato
y_ds = zeros(ceil((sample_length)/downsample_array(s)), n_antennas); 

% genero l'uscita dell'antenna y e la sua versione sotto campionata
noise = randn(sample_length, 1)*sqrt(noise_variance);
noise_j = randn(sample_length, 1)*sqrt(noise_variance);

target = (x_array(s,:))' + noise;
jammer = sin(2*pi*sin_jammer_freq(j)*(t')+2*pi*rand*enable_jammer_jitter) + noise_j;
y = collector([target,jammer],[incidentAngle,incidentAngle_j]); %calcolo l'uscita delle antenne della chirp che incide con l'angolo definito da incident angle

for p = 1 : n_antennas
    y_ds(:,p) = decimate(y(:,p), downsample_array(s), 128, "fir"); 
end

beamformer = phased.SubbandPhaseShiftBeamformer('SensorArray',myArray, ...
    'Direction',incidentAngle,'OperatingFrequency',fc, ...
    'SampleRate',CollectorSampleRate, 'NumSubbands', 100, 'SubbandsOutputPort',true, ...
    'WeightsOutputPort',true);
[z,w,subbandfreq] = beamformer(y);

pattern(myArray,subbandfreq(1:10).',[-180:180],0, ...
    'CoordinateSystem','rectangular','Weights',w(:,1:10))
legend('location','SouthEast')
%%
z_target = beamformer(y);

z_jammer = beamformer(y);



%plot del segnale x, y e y sottocampionata

f_2GHz = (1:sample_length)*1/sample_width;
f_ds   = f_2GHz(1:length(y_ds));

figure(1)
plot(f_2GHz,20*log10(abs(fft(target))/max(abs(fft(target)))))
xlabel('f [Hz]')
ylabel('x [dB]')

figure(2)
spectrogram(target,[],[],[],"centered")

figure(3)
plot(f_2GHz,20*log10(abs(fft(jammer))/max(abs(fft(jammer)))))
xlabel('f [Hz]')
ylabel('x [dB]')

figure(4)
spectrogram(jammer,[],[],[],"centered")

figure(5)
plot(f_2GHz,20*log10(abs(fft(y(:,1)))/max(abs(fft(y(:,1))))))
xlabel('f [Hz]')
ylabel('x [dB]')

figure(6)
spectrogram(y(:,1),[],[],[],"centered")

figure(7)
plot(f_2GHz,20*log10(abs(fft(z_target(:,1)))/max(abs(fft(z_target(:,1))))))
xlabel('f [Hz]')
ylabel('y [dB]')

figure(8)
spectrogram(z_target(:,1),[],[],[],"centered")

figure(9)
plot(f_2GHz,20*log10(abs(fft(z_jammer(:,1)))/max(abs(fft(z_jammer(:,1))))))
xlabel('f [Hz]')
ylabel('y [dB]')

figure(10)
spectrogram(z_jammer(:,1),[],[],[],"centered")

%%
pattern(myArray,fc,[-180:180],0,'PropagationSpeed',3e8,'CoordinateSystem','rectangular','Type','powerdb')