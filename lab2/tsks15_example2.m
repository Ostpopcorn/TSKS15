[melodies, ids, mismatches] = SignalGenerator(2, 100, 3);
melody = melodies(:, 1);
sampling_frequency = 8820;
soundsc(melody, sampling_frequency)
audiowrite('example.wav', melody / max(abs(melody)), sampling_frequency)