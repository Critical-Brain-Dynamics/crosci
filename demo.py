# This file is part of crosci, licensed under the Academic Public License.
# See LICENSE.txt for more details.

import os

import numpy as np
from crosci.biomarkers import (
    compute_spectrum_biomarkers,
    compute_band_biomarkers,
    get_frequency_bins,
)
from matplotlib import pyplot as plt
from time import time

# Get the full path of the current script
script_path = os.path.abspath(__file__)
# Get the directory of the current script
script_dir = os.path.dirname(script_path)
# Here I define the crosci path as the parent of the directory that contains demo.
# If you would like to move the demo script to a different path, make sure that crosci_code_path is set to the path
# where crosci is located.


def generate_signal_and_run(runtime="c"):
    print("Running with runtime", runtime)
    # define random signal with specified number of channels to test the code
    # note for that because this is a random signal, most DFA values will be <0.6, and so in these cases
    # fEI will not be computed (it will have a value of NaN)
    # this isgnal can be replaced by any EEG/MEG files. you can load these files using mne (https://mne.tools/stable/index.html)
    # make sure to feed only the signal matrix to the DFA&fEI algorithms
    sampling_frequency = 250
    num_seconds = 180
    num_channels = 128
    signal = np.random.rand(num_channels, num_seconds * sampling_frequency)

    frequency_range = [8, 16]
    # compute DFA and fEI across the spectrum
    biomarkers = compute_band_biomarkers(
        signal, sampling_frequency, frequency_range, runtime=runtime
    )
    DFA_array = biomarkers["DFA"]
    fEI_array = biomarkers["fEI"]
    print("DFA array for frequency range", frequency_range, DFA_array)
    print("fEI array for frequency range", frequency_range, fEI_array)

    spectrum_range = [1, 45]
    # compute DFA and fEI across the spectrum
    biomarkers = compute_spectrum_biomarkers(
        signal, sampling_frequency, spectrum_range, runtime=runtime
    )
    DFA_matrix = biomarkers["DFA"]
    fEI_matrix = biomarkers["fEI"]

    ############# PLOTTING OF RESULTS
    # idx of channel to plot
    chan_to_plot = 0

    # get all the frequency bins for which DFA/fEI were computed
    frequency_bins = get_frequency_bins(spectrum_range)

    # define the
    x_labels = []
    for i in range(len(frequency_bins)):
        x_labels.append(
            str(round(frequency_bins[i][0], 1))
            + "-"
            + str(round(frequency_bins[i][1], 1))
        )
    x_ticks = np.arange(len(frequency_bins))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(x_ticks, DFA_matrix[chan_to_plot, :])
    ax[0].set_xlabel("Frequency (Hz)")
    ax[0].set_ylabel("DFA")
    ax[0].set_xticks(x_ticks, x_labels, rotation=45)

    # note that fE/I will be NaN in this case, so nothing will be swhon
    ax[1].plot(x_ticks, fEI_matrix[chan_to_plot, :])
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("fEI")
    ax[1].set_xticklabels(x_labels)
    ax[1].set_xticks(x_ticks, x_labels, rotation=45)


if __name__ == "__main__":
    start_cpython = time()
    generate_signal_and_run(runtime="c")
    end_cpython = time()

    start_python_only = time()
    generate_signal_and_run(runtime="python")
    end_python_only = time()

    print("time cpython", end_cpython - start_cpython)
    print("time python only", end_python_only - start_python_only)
