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
pitch_offsets = [0.975, 1.025]

# %%
"""
Generate H_i for all twelve notes.
All f_i exists in sg.dict_note2frequency
"""
# Contains { 0.975 = {"C" = H-matix, ....}
#            1.025 = {"C" = H-matix, ....} }
h_matrices_single_note = {}
h_matrices_triple_note = {}

for pitch_offset in pitch_offsets:
    h_matrices_single_note[pitch_offset] = {}
    current_h_matrices_single_note = h_matrices_single_note[pitch_offset]

    for note_name, note_freq in sg.dict_note2frequency.items():
        current_h_matrices_single_note[note_name] = np.zeros((k, 2))
        current_h_matrices_single_note[note_name][:, 0] = np.cos(2 * np.pi * note_freq * pitch_offset * n / fs)
        current_h_matrices_single_note[note_name][:, 1] = np.sin(2 * np.pi * note_freq * pitch_offset * n / fs)

    h_matrices_triple_note[pitch_offset] = {}
    current_h_matrices_triple_note = h_matrices_triple_note[pitch_offset]

    for note_name, note_freq in sg.dict_note2frequency.items():
        current_h_matrices_triple_note[note_name] = np.zeros((k, 6))
        current_h_matrices_triple_note[note_name][:, 0] = np.cos(2 * np.pi * note_freq * pitch_offset * 1 * n / fs)
        current_h_matrices_triple_note[note_name][:, 1] = np.sin(2 * np.pi * note_freq * pitch_offset * 1 * n / fs)
        current_h_matrices_triple_note[note_name][:, 2] = np.cos(2 * np.pi * note_freq * pitch_offset * 3 * n / fs)
        current_h_matrices_triple_note[note_name][:, 3] = np.sin(2 * np.pi * note_freq * pitch_offset * 3 * n / fs)
        current_h_matrices_triple_note[note_name][:, 4] = np.cos(2 * np.pi * note_freq * pitch_offset * 5 * n / fs)
        current_h_matrices_triple_note[note_name][:, 5] = np.sin(2 * np.pi * note_freq * pitch_offset * 5 * n / fs)

# %%
""" 
Task one, single tone generator
"""
print("start part one")


def note_detector(note, h_matrices):
    h_values = {}
    max_notes_per_pitch = {}
    for current_pitch_offset in pitch_offsets:
        h_values[current_pitch_offset] = {}
        current_h_values = h_values[current_pitch_offset]
        for current_note_name, h_matrix in h_matrices[current_pitch_offset].items():  # h_matrix) **2
            current_h_values[current_note_name] = np.linalg.norm(np.matmul(np.transpose(tone), h_matrix)) ** 2
        current_max_note = max(current_h_values, key=current_h_values.get)
        max_notes_per_pitch[current_pitch_offset] = current_max_note
    pitch = max(max_notes_per_pitch, key=lambda val: h_values[val][max_notes_per_pitch[val]])
    print("Pitch", pitch, "Note", max_notes_per_pitch[pitch])
    max_note = max_notes_per_pitch[pitch]
    return max_note, pitch, h_values


# Contains {"C",||y_n^T*H_j||^2}
max_note, pitch, returned_h_values = note_detector(tone, h_matrices_single_note)
print("Tone is: ", max_note, "with power", returned_h_values[pitch][max_note])
plt.figure()
plt.plot(np.arange(len(returned_h_values[pitch])), returned_h_values[pitch].values())
plt.xlabel('H')
plt.ylabel('magnitude')
plt.savefig('tone_many_j.png')
print("end task one")

# %%
