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
from tqdm import tqdm

sg = SignalGenerator()
melody, idx, mismatch = sg.generate_random_melody(100, 1)
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
tone_length = len(tone)  # length of one note, k in book
n = np.arange(tone_length)  # array [0, ... ,tone_length-1]
pitch_offsets = [0.975, 1.025]

"""
Generate H_i for all twelve notes.
All f_i exists in sg.dict_note2frequency
"""
# Contains { 0.975 = {"C" = H-matrix, ....}
#            1.025 = {"C" = H-matrix, ....} }
h_matrices_single_note = {}
h_matrices_triple_note = {}

for pitch_offset in pitch_offsets:
    h_matrices_single_note[pitch_offset] = {}
    current_h_matrices_single_note = h_matrices_single_note[pitch_offset]

    for note_name, note_freq in sg.dict_note2frequency.items():
        current_h_matrices_single_note[note_name] = np.zeros((tone_length, 2))
        current_h_matrices_single_note[note_name][:, 0] = np.cos(2 * np.pi * note_freq * pitch_offset * n / fs)
        current_h_matrices_single_note[note_name][:, 1] = np.sin(2 * np.pi * note_freq * pitch_offset * n / fs)

    h_matrices_triple_note[pitch_offset] = {}
    current_h_matrices_triple_note = h_matrices_triple_note[pitch_offset]

    for note_name, note_freq in sg.dict_note2frequency.items():
        current_h_matrices_triple_note[note_name] = np.zeros((tone_length, 6))
        current_h_matrices_triple_note[note_name][:, 0] = np.cos(2 * np.pi * note_freq * pitch_offset * 1 * n / fs)
        current_h_matrices_triple_note[note_name][:, 1] = np.sin(2 * np.pi * note_freq * pitch_offset * 1 * n / fs)
        current_h_matrices_triple_note[note_name][:, 2] = np.cos(2 * np.pi * note_freq * pitch_offset * 3 * n / fs)
        current_h_matrices_triple_note[note_name][:, 3] = np.sin(2 * np.pi * note_freq * pitch_offset * 3 * n / fs)
        current_h_matrices_triple_note[note_name][:, 4] = np.cos(2 * np.pi * note_freq * pitch_offset * 5 * n / fs)
        current_h_matrices_triple_note[note_name][:, 5] = np.sin(2 * np.pi * note_freq * pitch_offset * 5 * n / fs)


def note_detector(tone, h_matrices):
    """
    Function that can classify signal by the given h_matrices
    :param tone: array containing the note to be classified:
    :param h_matrices:  a dict for different h_matrices depending on pitch offset.
    Contains { 0.975 = {"C" = H-matrix, ....}
               1.025 = {"C" = H-matrix, ....} }:
    :return note as string, pitch offset, h_values for all pitches:
    """
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
    # print("Pitch", pitch, "Note", max_notes_per_pitch[pitch])
    max_note = max_notes_per_pitch[pitch]
    return max_note, pitch, h_values


def song_checker(detector_notes, melody_index, pitch):
    correct_notes = sg.melodies[int(melody_index)]
    count = 0
    for index in range(nr_tones):
        if correct_notes[index] != detector_notes[index][0] or detector_notes[index][1] != pitch:
            count += 1

    return count


def song_detector(melody, melody_index, pitch, number_of_notes):
    # Ta fram alla tolv via detektorn
    tone_len = int(nr_samples / nr_tones)
    detector_notes = [["", 0] for i in range(12)]
    for i in range(nr_tones):
        # tone = melody[:int(nr_samples / nr_tones)]
        # I really want to make it one index shorter so no index i counted twice
        current_tone = melody[i * tone_len:(i + 1) * tone_len]
        if number_of_notes == 1:
            b = note_detector(current_tone, h_matrices_single_note)
            detector_notes[i][0] = b[0]
            detector_notes[i][1] = b[1]
        elif number_of_notes == 3:
            detector_notes[i][0], detector_notes[i][1], b = note_detector(current_tone, h_matrices_triple_note)
        else:
            print("PANIK")
    return song_checker(detector_notes, melody_index, pitch)


# %%
""" 
Task one, single tone generator
"""

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
"""
MonteCarlo that shit!
"""
number_of_notes_generator = [1, 3, 1, 3]
number_of_notes_classifier = [1, 1, 3, 3]

number_of_monte_carlo_runs = 20
snr_values = np.arange(-50, 0, 2)
sigma2_value = 10 ** (-snr_values / 10)
error_counter = [np.zeros(len(snr_values)) for i in range(4)]
# for SNR_index in range(len(snr_values)):
for i in range(4):
    print("G", number_of_notes_generator[i], "C", number_of_notes_classifier[i])
    for SNR_index in tqdm(range(len(snr_values))):
        melodies, ids, pitches = sg.generate_random_melodies(number_of_monte_carlo_runs, snr_values[SNR_index],
                                                             number_of_notes_generator[i])
        for run_no in range(number_of_monte_carlo_runs):
            error_counter[i][SNR_index] += song_detector(melodies[:, run_no], ids[run_no], pitches[run_no],
                                                         number_of_notes_classifier[i])

# %%
f4, axs = plt.subplots(1)

# plt.figure(1)
g1c1_plot,  = axs.plot(snr_values, error_counter[0] / (nr_tones * number_of_monte_carlo_runs))
g3c1_plot,  = axs.plot(snr_values, error_counter[1] / (nr_tones * number_of_monte_carlo_runs))
g1c3_plot,  = axs.plot(snr_values, error_counter[2] / (nr_tones * number_of_monte_carlo_runs))
g3c3_plot,  = axs.plot(snr_values, error_counter[3] / (nr_tones * number_of_monte_carlo_runs))
plt.xlabel('SNR')
plt.ylabel('Number of errors')
plt.savefig('error_plot.png')
plt.legend((g1c1_plot, g3c1_plot, g1c3_plot, g3c3_plot), ("G1C1", "G3C1", "´G1C3", "G3C3"))
plt.show()
