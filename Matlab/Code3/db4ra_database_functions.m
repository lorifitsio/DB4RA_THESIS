	
function [signals, y_ds_cell] = db4ra_database_functions (db4ra_collector, f_stop_array,n_dataset_samples_woj, azimuth_angle_vector, elevation_angle, noise_variance, window_length, n_scenarios, sample_length, x_array, n_antennas, fir_coeffs, downsample_array)

	y_ds_cell = zeros(n_dataset_samples_woj, 3);
	signals = zeros (window_length/downsample_array(1),n_antennas*2,1,n_dataset_samples_woj);
	
    parfor m = 1 : n_dataset_samples_woj
        azimuth_angle = azimuth_angle_vector(m); %imposto l'angolo di partenza per la chirp in questione
        %spazzo tutto il range di angoli
		incidentAngle = [azimuth_angle;elevation_angle];
		noise = randn(window_length, 1)*sqrt(noise_variance); %genero un vettore di rumore per ogni iterazione
		
		chirp_sel = randi([1 n_scenarios], 1, 1);
        window_sel = randi([1 sample_length], 1, 1);
			
		y = db4ra_collector((x_array(chirp_sel,window_sel:window_sel+window_length-1))' + noise,incidentAngle); %calcolo l'uscita delle antenne della chirp che incide con l'angolo definito da incident angle
		
        y_ds_temp = zeros (window_length/downsample_array(1),n_antennas);

		%ciclo per il numero di antenne, poiché y è un array di 4 vettori, uno per ciascuna antenna
		for p = 1 : n_antennas
		 	temp = downsample(conv(y(:,p),fir_coeffs(chirp_sel,:),'same'), downsample_array(1)); %effettuo la decimazione
		 	y_ds_temp(:,p) = temp; %salvo il l'uscita dell'antenna decimata nel dataset
        end
        
    	% temp = downsample(conv(y(:,1),fir_coeffs(chirp_sel,:),'same'), downsample_array(1)); %effettuo la decimazione
		% y_ds_temp(:,1) = temp; %salvo il l'uscita dell'antenna decimata nel dataset
        % 
        % temp = downsample(conv(y(:,2),fir_coeffs(chirp_sel,:),'same'), downsample_array(1)); %effettuo la decimazione
		% y_ds_temp(:,2) = temp; %salvo il l'uscita dell'antenna decimata nel dataset
        % 
        % 
        % temp = downsample(conv(y(:,3),fir_coeffs(chirp_sel,:),'same'), downsample_array(1)); %effettuo la decimazione
		% y_ds_temp(:,3) = temp; %salvo il l'uscita dell'antenna decimata nel dataset
        % 
        % 
        % temp = downsample(conv(y(:,4),fir_coeffs(chirp_sel,:),'same'), downsample_array(1)); %effettuo la decimazione
		% y_ds_temp(:,4) = temp; %salvo il l'uscita dell'antenna decimata nel dataset
	
	
		signals(:,:,1,m)=[real(y_ds_temp) imag(y_ds_temp)];

		y_ds_cell(m,:) = [azimuth_angle f_stop_array(chirp_sel) 0];
    end
end