
function [signals, y_ds_cell, doa, jam] = db4ra_database_sweptcw_jammer (sweptcw,remove_signal, sweptcw_window_Sel, db4ra_collector, azimuth_start_j, t2j_distance, azimuth_stop_j, f_stop_array,n_chirp_angles, n_jammer_angles_per_chirp, azimuth_angle_vector, elevation_angle, elevation_angle_j, noise_variance, window_length, n_scenarios, sample_length, x_array, n_antennas, fir_coeffs, downsample_array)

    n_dataset_samples_wj = n_chirp_angles*n_jammer_angles_per_chirp;
    y_ds_cell = zeros(n_dataset_samples_wj, 3);
	signals = zeros (window_length/downsample_array(1),n_antennas*2,1,n_dataset_samples_wj);
    doa = zeros(n_dataset_samples_wj, 1);
    jam = zeros(n_dataset_samples_wj, 1);

    signals_temp = zeros (window_length/downsample_array(1),n_antennas*2,1,n_jammer_angles_per_chirp);

    y_ds_cell_temp = zeros(n_jammer_angles_per_chirp, 3);
	doa_temp  = zeros(n_jammer_angles_per_chirp, 1);
	jam_temp = zeros(n_jammer_angles_per_chirp, 1);

    y_ds_row = 1;
    
    for m = 1 : n_chirp_angles
            
		azimuth_angle = azimuth_angle_vector(m); %imposto l'angolo di partenza per la chirp in questione

		azimuth_angle_vector_j = randi([azimuth_start_j+t2j_distance azimuth_stop_j-t2j_distance-1], 1, n_jammer_angles_per_chirp);

		sel = azimuth_angle_vector(m) > azimuth_angle_vector_j;
		azimuth_angle_vector_j(sel) = azimuth_angle_vector_j(sel) - t2j_distance;
		
		sel = azimuth_angle_vector(m) <= azimuth_angle_vector_j;
		azimuth_angle_vector_j(sel) = azimuth_angle_vector_j(sel) + t2j_distance+1;
        
        window_sel = randi([1 sample_length], 1, 1);
		    
        chirp_sel = randi([1 n_scenarios], 1, 1);
        sweptc_cw_window = randi([1 sweptcw_window_Sel(chirp_sel)], 1, 1);

        %spazzo tutto il range di angoli del segnale utile
		parfor i = 1 : n_jammer_angles_per_chirp
			incidentAngle = [azimuth_angle;elevation_angle];
			azimuth_angle_j = azimuth_angle_vector_j(i);
			
		    %spazzo tutto il range di angoli del segnale interferente
				
			incidentAngle_j = [azimuth_angle_j;elevation_angle_j];
		    noise = randn(window_length, 1)*sqrt(noise_variance); %genero un vettore di rumore per ogni iterazione
			
                    
            noise_j = randn(window_length, 1)*sqrt(noise_variance);
											
			target = (x_array(chirp_sel,window_sel:window_sel+window_length-1))' + noise;
        												  
			jammer = sweptcw(sweptc_cw_window:sweptc_cw_window+window_length-1)' + noise_j;

            if remove_signal == true 
    			y = db4ra_collector(jammer,incidentAngle_j); %calcolo l'uscita delle antenne della chirp che incide con l'angolo definito da incident angle
            else
    			y = db4ra_collector([target,jammer],[incidentAngle,incidentAngle_j]); %calcolo l'uscita delle antenne della chirp che incide con l'angolo definito da incident angle
            end
            y_ds_temp = zeros (window_length/downsample_array(1),n_antennas);

			%ciclo per il numero di antenne, poiché y è un array di 4 vettori, uno per ciascuna antenna
			for p = 1 : n_antennas
				temp = downsample(conv(y(:,p),fir_coeffs(chirp_sel,:),'same'), downsample_array(1)); %effettuo la decimazione
				y_ds_temp(:,p) = temp; %salvo il l'uscita dell'antenna decimata nel dataset
			end

            signals_temp(:,:,1,i)=[real(y_ds_temp) imag(y_ds_temp)];

            y_ds_cell_temp(i,:) = [azimuth_angle f_stop_array(chirp_sel) sweptc_cw_window];
        	doa_temp(i) = azimuth_angle_j;
			jam_temp(i) = 1; 
        end

        signals(:,:,1,1+(m-1)*n_jammer_angles_per_chirp:m*n_jammer_angles_per_chirp)=signals_temp;

        y_ds_cell(1+(m-1)*n_jammer_angles_per_chirp:m*n_jammer_angles_per_chirp,:) = y_ds_cell_temp;
    	doa((1+(m-1)*n_jammer_angles_per_chirp:m*n_jammer_angles_per_chirp)) = doa_temp;
		jam((1+(m-1)*n_jammer_angles_per_chirp:m*n_jammer_angles_per_chirp)) = jam_temp; 
    end