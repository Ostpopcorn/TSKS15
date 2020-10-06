% Copyright (C) 2018 Daniel Verenzuela
% SignalGenerator for Matlab.
function [signal, melody_id, freq_offset] =  SignalGeneratorDV(Nr_MC, SNR_dB, Nr_tones)

note_duration = 0.4;
pause_duration = 0.01;
sampling_freq = 8820;

Nr_samples_note = floor(note_duration*sampling_freq);
Nr_pause_note = floor(pause_duration*sampling_freq);

noise_std = 1./(sqrt(   10.^(SNR_dB./10)   ));




notes_char = {'C@','C#@','D@','D#@','E@','F@','F#@','G@','G#@','A@','A#@','B@'};
notes_freq_1 = 440.*(2.^((-9:2)./12));
notes_txt = {...
    'C4', 'C4', 'G4', 'G4', 'A4', 'A4', 'G4', 'G4', 'F4', 'F4', 'E4', 'E4';...
    'G4', 'G4', 'A4', 'E4', 'G4', 'G4', 'C5', 'D5', 'E5', 'E5', 'E5', 'D5';...
    'G3', 'B3', 'D4', 'B4', 'G4', 'D4', 'G3', 'C4', 'E4', 'C5', 'G4', 'E4';...
    'G4', 'G4', 'G4', 'B4', 'A4', 'A4', 'A4', 'C5', 'B4', 'G4', 'A4', 'F#4';...
    'A3', 'D4', 'A3', 'E4', 'A3', 'F4', 'A3', 'G4', 'A3', 'F4', 'A3', 'E4';...
    'B3', 'D#4', 'G4', 'D#5', 'B4', 'G4', 'C4', 'E4', 'G4', 'E5', 'C5', 'G4';...
    'D3', 'A3', 'C4', 'F#4', 'D4', 'C4', 'D3', 'G3', 'C4', 'E4', 'C4', 'G3';...
    'E4', 'E4', 'F4', 'G4', 'G4', 'F4', 'E4', 'D4', 'C4', 'C4', 'D4', 'E4';...
    'E4', 'F4', 'G4', 'E4', 'G4', 'E4', 'G4', 'E4', 'G4', 'D5', 'C5', 'E4';...
    'E5', 'D#5', 'E5', 'D#5', 'E5', 'B4', 'D5', 'C5', 'A4', 'E3', 'A3', 'C4'};



[N_melodies,N_notes_per_melody] = size(notes_txt);


freq_notes = zeros([N_melodies,N_notes_per_melody]);

for n_scale = 1:8
    notes_char_t = strrep(notes_char,'@',num2str(n_scale));
    for i_notes = 1:12
        ind  = strfind(notes_txt,notes_char_t{i_notes});
        freq_notes((~cellfun('isempty', ind))) = notes_freq_1(i_notes).*2.^(n_scale - 4 );
    end
end


melody_id= randi(N_melodies,1,Nr_MC);
freq_notes = freq_notes(melody_id,:);
freq_offset = [0.975, 1.025];
freq_offset = freq_offset(randi(2,1,Nr_MC));

total_nr_samples =   (Nr_samples_note+Nr_pause_note)*N_notes_per_melody;
signal = zeros(total_nr_samples,Nr_MC);

samples_per_note = (0:(Nr_samples_note-1)).'./sampling_freq;
amplitude = ones(Nr_MC,N_notes_per_melody);
ind_amplitude = 3.*ones(Nr_MC,1);%randi(3,Nr_MC,1);
amplitude(ind_amplitude == 1,:) = exprnd(1,[nnz(ind_amplitude == 1),N_notes_per_melody]);
amplitude(ind_amplitude == 2,:) = ones(nnz(ind_amplitude == 2),1)*(0.8 + 0.4*(1:12)./12);
amplitude(ind_amplitude == 3,:) = rand([nnz(ind_amplitude == 3),N_notes_per_melody]).*0.4 + 0.8;
disp(['Generating ',num2str(Nr_MC),' random melodies with SNR ',num2str(SNR_dB),' [dB] and ',num2str(Nr_tones),' tones'])
for i_mc = 1:Nr_MC
    
    if mod(i_mc,ceil(Nr_MC./10)) == 0
        disp([' nr MC : ',num2str(i_mc),' of ',num2str(Nr_MC),''])
    end
    t = freq_offset(i_mc).*samples_per_note*ones(1,N_notes_per_melody)*diag(freq_notes(i_mc,:));
    phase_n = rand(3,N_notes_per_melody);
    if Nr_tones == 1
        signal_t = [( cos(2.*pi.*t + 2.*pi.*ones(Nr_samples_note,1)*phase_n(1,:) ) )*diag(amplitude(i_mc,:)); zeros(Nr_pause_note,N_notes_per_melody) ] ;        
    elseif Nr_tones == 3
        signal_t = [(1./sqrt(3)).*( cos(2.*pi.*t + 2.*pi.*ones(Nr_samples_note,1)*phase_n(1,:))...
                                    +cos(2.*3.*pi.*t + 2.*pi.*ones(Nr_samples_note,1)*phase_n(2,:))...
                                    +cos(2.*5.*pi.*t + 2.*pi.*ones(Nr_samples_note,1)*phase_n(3,:) ) )*diag(amplitude(i_mc,:)); zeros(Nr_pause_note,N_notes_per_melody) ] ;        
                                
    end
    signal(:,i_mc) = reshape(signal_t, total_nr_samples,1 ) + noise_std.*randn(total_nr_samples,1);
end