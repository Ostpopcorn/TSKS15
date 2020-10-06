% Note: When using Octave/Matlab, it may be beneficial for you to generate several melodies with each call to SignalGenerator. This is because the call invokes the 'system', so the overhead is a bit longer than when using Python.

%% Example: Spectrum of the first note
%Generate 2 melodies, with SNR 100 dB, and 3 tones.
[melodies, ids, mismatches] = SignalGenerator(2, 100, 3);
% melodies: a matrix containing the received (sampled) signal. Each column is one melody
% ids: a vector with the intended(true) melody ids (from 1 to 10)
% mismatches: a vector with the pitch mismatches of the generator. (for example [1.025, 0.975])
melody = melodies(:,1);
nr_samples = length(melody);
nr_tones = 12;
tone = melody(1:nr_samples/nr_tones);
nr_tone_samples = length(tone);
sampling_frequency = 8820;
freqs = (1:nr_tone_samples) * sampling_frequency / nr_tone_samples;
figure()
spectrum = abs(fft(tone));
plot(freqs(1:(nr_tone_samples/2)), spectrum(1:(nr_tone_samples/2)))
xlabel('frequency [Hz]')
ylabel('magnitude')
print('matlab-example1','-dpng')
