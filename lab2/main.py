#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: adrian
"""

# Imports
from lab2.SignalGenerator import SignalGenerator
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

sg = SignalGenerator()
melody, idx, mismatch = sg.generate_random_melody(100, 3)
"""
Get a random note 
"""
nr_samples = len(melody)
nr_tones = 12  # all melodies have 12 tones
fs = sg.sampling_frequency
tone = melody[:int(nr_samples / nr_tones)]
print("Melody id:", idx, ", Pitch:", mismatch)
a = rnd.standard_exponential(1)
# phi = rnd.uniform(-np.pi, np.pi, 1) # Is in the randomly generated melody
SNR = 10
k = len(tone)  # length of one note
n = np.arange(k)  # array [0, ... ,k-1]

# %%
"""
Generate H_i for all twelve notes.
All f_i exists in sg.dict_note2frequency
"""
# Contains {"C",H-matix}
h_matrices_single_note = {}

for note_name, note_freq in sg.dict_note2frequency.items():
    h_matrices_single_note[note_name] = np.zeros((k, 2))
    h_matrices_single_note[note_name][:, 0] = np.cos(2 * np.pi * note_freq * n / fs)
    h_matrices_single_note[note_name][:, 1] = np.sin(2 * np.pi * note_freq * n / fs)

h_matrices_triple_note = {}

for note_name, note_freq in sg.dict_note2frequency.items():
    h_matrices_triple_note[note_name] = np.zeros((k, 6))
    h_matrices_triple_note[note_name][:, 0] = np.cos(2 * np.pi * note_freq * 1 * n / fs + phi)
    h_matrices_triple_note[note_name][:, 1] = np.sin(2 * np.pi * note_freq * 1 * n / fs + phi)
    h_matrices_triple_note[note_name][:, 2] = np.cos(2 * np.pi * note_freq * 3 * n / fs + phi)
    h_matrices_triple_note[note_name][:, 3] = np.sin(2 * np.pi * note_freq * 3 * n / fs + phi)
    h_matrices_triple_note[note_name][:, 4] = np.cos(2 * np.pi * note_freq * 5 * n / fs + phi)
    h_matrices_triple_note[note_name][:, 5] = np.sin(2 * np.pi * note_freq * 5 * n / fs + phi)

# %%
""" 
Task one, single tone generator
"""
print("start part one")


def note_detector(note):
    h_value = {}
    for current_note_name, h_matrix in h_matrices_single_note.items():  # h_matrix) **2
        h_value[current_note_name] = np.linalg.norm(np.matmul(np.transpose(tone), h_matrix)) ** 2
    max_note = max(h_value, key=h_value.get)
    return max_note, h_value


# Contains {"C",||y_n^T*H_j||^2}
max_note, h_values = (note_detector(tone))
print("Tone is: ", max_note, "with power", h_values[max_note])
plt.figure()
plt.plot(np.arange(len(h_values)), h_values.values())
plt.xlabel('H')
plt.ylabel('magnitude')
plt.savefig('tone_many_j.png')
print("end task one")

# %%

