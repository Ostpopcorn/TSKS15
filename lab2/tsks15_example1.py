#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:40:32 2018
Example 1: Spectrum of the first note of a random melody
@author: marcus
"""

# Imports
from lab2.SignalGenerator import SignalGenerator
import numpy as np
import matplotlib.pyplot as plt

# Program
sg = SignalGenerator()
# generate a random melody, with SNR 100 dB, and 3 tones
melody, idx, mismatch = sg.generate_random_melody(100, 3)
nr_samples = len(melody)
nr_tones = 12  # all melodies have 12 tones
tone = melody[:int(nr_samples/nr_tones)]
nr_tone_samples = len(tone)
spectrum = np.abs(np.fft.fft(tone))
fs = sg.sampling_frequency
freqs = np.arange(nr_tone_samples) * fs / nr_tone_samples
plt.figure()
plt.plot(freqs[:int(nr_tone_samples/2)], spectrum[:int(nr_tone_samples/2)])
plt.xlabel('frequency [Hz]')
plt.ylabel('magnitude')
plt.savefig('python-example1.eps')
