# This file is part of crosci, licensed under the Academic Public License.
# See LICENSE.txt for more details.

import numpy as np
from numpy.matlib import repmat
from .run_DFA import run_DFA
from .run_fEI import run_fEI

from .outliers import generalizedESD
import multiprocessing
import mne
from mne.filter import next_fast_len
from scipy.signal import hilbert
from joblib import Parallel, delayed


def fEI(
    signal,
    sampling_frequency,
    window_size_sec,
    window_overlap,
    DFA_array,
    runtime="c",
    bad_idxes=[],
):
    """Calculates fEI (on a set window size) for signal

        Steps refer to description of fEI algorithm in Figure 2D of paper:
          Measurement of excitation inhibition ratio in autism spectrum disorder using critical brain dynamics
          Scientific Reports (2020)
          Hilgo Bruining*, Richard Hardstone*, Erika L. Juarez-Martinez*, Jan Sprengers*, Arthur-Ervin Avramiea, Sonja Simpraga, Simon J. Houtman, Simon-Shlomo Poil5,
          Eva Dallares, Satu Palva, Bob Oranje, J. Matias Palva, Huibert D. Mansvelder & Klaus Linkenkaer-Hansen
          (*Joint First Author)

        Originally created by Richard Hardstone (2020), rhardstone@gmail.com
        Please note that commercial use of this algorithm is protected by Patent claim (PCT/NL2019/050167) “runtime of determining brain activity”; with priority date 16 March 2018

    Parameters
    ----------
    signal: array, shape(n_channels,n_times)
        amplitude envelope for all channels
    sampling_frequency: integer
        sampling frequency of the signal
    window_size_sec: float
        window size in seconds
    window_overlap: float
        fraction of overlap between windows (0-1)
    DFA_array: array, shape(n_channels)
        array of DFA values, with corresponding value for each channel, used for thresholding fEI
    runtime: string
        can take either value 'c' or 'python'. note that 'c' code is 10 times faster
    bad_idxes: array, shape(n_channels)
        channels to ignore from computation are marked with 1, the rest with 0. can also be empty list,
        case in which all channels are computed

    Returns
    -------
    fEI_outliers_removed: array, shape(n_channels)
        fEI values, with outliers removed
    fEI_val: array, shape(n_channels)
        fEI values, with outliers included
    num_outliers: integer
        number of detected outliers
    wAmp: array, shape(n_channels, num_windows)
        windowed amplitude, computed across all channels/windows
    wDNF: array, shape(n_channels, num_windows)
        windowed detrended normalized fluctuation, computed across all channels/windows
    """

    window_size = int(window_size_sec * sampling_frequency)

    num_chans = np.shape(signal)[0]
    length_signal = np.shape(signal)[1]

    channels_to_ignore = [False] * num_chans

    for bad_idx in bad_idxes:
        channels_to_ignore[bad_idx] = True

    window_offset = int(np.floor(window_size * (1 - window_overlap)))
    all_window_index = _create_window_indices(length_signal, window_size, window_offset)
    num_windows = np.shape(all_window_index)[0]

    fEI_val = np.zeros((num_chans, 1))
    fEI_val[:] = np.NAN
    fEI_outliers_removed = np.zeros((num_chans, 1))
    fEI_outliers_removed[:] = np.NAN
    num_outliers = np.zeros((num_chans, 1))
    num_outliers[:] = np.NAN
    wAmp = np.zeros((num_chans, num_windows))
    wAmp[:] = np.NAN
    wDNF = np.zeros((num_chans, num_windows))
    wDNF[:] = np.NAN

    for ch_idx in range(num_chans):
        if channels_to_ignore[ch_idx]:
            continue

        original_amp = signal[ch_idx, :]

        if np.min(original_amp) == np.max(original_amp):
            print("Problem computing fEI for channel idx " + str(ch_idx))
            continue

        if runtime == "c":
            (w_original_amp, w_detrendedNormalizedFluctuations) = run_fEI(
                original_amp, sampling_frequency, window_size_sec, window_overlap
            )
        elif runtime == "python":
            signal_profile = np.cumsum(original_amp - np.mean(original_amp))
            w_original_amp = np.mean(original_amp[all_window_index], axis=1)

            x_amp = repmat(np.transpose(w_original_amp[np.newaxis, :]), 1, window_size)
            x_signal = signal_profile[all_window_index]
            x_signal = np.divide(x_signal, x_amp)

            # Calculate local trend, as the line of best fit within the time window
            _, fluc, _, _, _ = np.polyfit(
                np.arange(window_size), np.transpose(x_signal), deg=1, full=True
            )
            # Convert to root-mean squared error, from squared error
            w_detrendedNormalizedFluctuations = np.sqrt(fluc / window_size)

        fEI_val[ch_idx] = (
            1 - np.corrcoef(w_original_amp, w_detrendedNormalizedFluctuations)[0, 1]
        )

        gesd_alpha = 0.05
        max_outliers_percentage = 0.025  # this is set to 0.025 per dimension (2-dim: wAmp and wDNF), so 0.05 is max
        # smallest value for max number of outliers is 2 for generalizedESD
        max_num_outliers = max(
            int(np.round(max_outliers_percentage * len(w_original_amp))), 2
        )
        outlier_indexes_wAmp = generalizedESD(
            w_original_amp, max_num_outliers, gesd_alpha
        )[1]
        outlier_indexes_wDNF = generalizedESD(
            w_detrendedNormalizedFluctuations, max_num_outliers, gesd_alpha
        )[1]
        outlier_union = outlier_indexes_wAmp + outlier_indexes_wDNF
        num_outliers[ch_idx, :] = len(outlier_union)
        not_outlier_both = np.setdiff1d(
            np.arange(len(w_original_amp)), np.array(outlier_union)
        )
        fEI_outliers_removed[ch_idx] = (
            1
            - np.corrcoef(
                w_original_amp[not_outlier_both],
                w_detrendedNormalizedFluctuations[not_outlier_both],
            )[0, 1]
        )

        wAmp[ch_idx, :] = w_original_amp
        wDNF[ch_idx, :] = w_detrendedNormalizedFluctuations

    fEI_val[DFA_array <= 0.6] = np.nan
    fEI_outliers_removed[DFA_array <= 0.6] = np.nan

    return (fEI_outliers_removed, fEI_val, num_outliers, wAmp, wDNF)


def DFA(
    signal,
    sampling_frequency,
    fit_interval,
    compute_interval,
    overlap=True,
    runtime="c",
    bad_idxes=[],
):
    """Calculates DFA of a signal

    Parameters
    ----------
    signal: array, shape(n_channels,n_times)
        amplitude envelope for all channels
    sampling_frequency: integer
        sampling frequency of the signal
    fit_interval: list, length 2
        interval (in seconds) over which the DFA exponent is fit. should be included in compute_interval
    compute_interval: list, length 2
        interval (in seconds) over which DFA is computed
    overlap: boolean
        if set to True, then windows are generated with an overlap of 50%
    runtime: string
        can take either value 'c' or 'python'. note that 'c' code is 10 times faster
    bad_idxes: array, shape(n_channels)
        channels to ignore from computation are marked with 1, the rest with 0. can also be empty list,
        case in which all channels are computed

    Returns
    -------
    dfa_array, window_sizes, fluctuations, dfa_intercept
    dfa_array: array, shape(n_channels)
        DFA value for each channel
    window_sizes: array, shape(num_windows)
        window sizes over which the fluctuation function is computed
    fluctuations: array, shape(num_windows)
        fluctuation function value at each computed window size
    dfa_intercept: array, shape(n_channels)
        DFA intercept for each channel
    """

    num_chans, num_timepoints = np.shape(signal)

    channels_to_ignore = [False] * num_chans
    for bad_idx in bad_idxes:
        channels_to_ignore[bad_idx] = True

    length_signal = np.shape(signal)[1]

    assert (
        fit_interval[0] >= compute_interval[0]
        and fit_interval[1] <= compute_interval[1]
    ), "CalcInterval should be included in ComputeInterval"
    assert compute_interval[0] >= 0.1 and compute_interval[1] <= 1000, (
        "ComputeInterval should be between 0.1 and 1000 seconds"
    )
    assert compute_interval[1] / sampling_frequency <= num_timepoints, (
        "ComputeInterval should not extend beyond the length of the signal"
    )

    # compute DFA window sizes for the given CalcInterval
    window_sizes = np.floor(np.logspace(-1, 3, 81) * sampling_frequency).astype(
        int
    )  # %logspace from 0.1 seccond (10^-1) to 1000 (10^3) seconds

    # make sure there are no duplicates after rounding
    window_sizes = np.sort(np.unique(window_sizes))

    window_sizes = window_sizes[
        (window_sizes >= compute_interval[0] * sampling_frequency)
        & (window_sizes <= compute_interval[1] * sampling_frequency)
    ]

    dfa_array = np.zeros(num_chans)
    dfa_array[:] = np.NAN
    dfa_intercept = np.zeros(num_chans)
    dfa_intercept[:] = np.NAN
    fluctuations = np.zeros((num_chans, len(window_sizes)))
    fluctuations[:] = np.NAN

    if max(window_sizes) <= num_timepoints:
        for ch_idx in range(num_chans):
            if channels_to_ignore[ch_idx]:
                continue

            signal_for_channel = signal[ch_idx, :]

            if runtime == "c":
                [window_sizes, fluctuations_channel] = run_DFA(
                    signal_for_channel - np.mean(signal_for_channel),
                    sampling_frequency,
                    overlap,
                    window_sizes,
                )
                fluctuations[ch_idx, :] = fluctuations_channel
            elif runtime == "python":
                for i_window_size in range(len(window_sizes)):
                    if overlap:
                        window_overlap = 0.5
                    else:
                        window_overlap = 0

                    window_size = window_sizes[i_window_size]
                    window_offset = np.floor(window_size * (1 - window_overlap))
                    all_window_index = _create_window_indices(
                        length_signal, window_sizes[i_window_size], window_offset
                    )
                    # First we convert the time series into a series of fluctuations y(i) around the mean.
                    demeaned_signal = signal_for_channel - np.mean(signal_for_channel)
                    # Then we integrate the above fluctuation time series ('y').
                    signal_profile = np.cumsum(demeaned_signal)

                    x_signal = signal_profile[all_window_index]

                    # Calculate local trend, as the line of best fit within the time window -> fluc is the sum of squared residuals
                    _, fluc, _, _, _ = np.polyfit(
                        np.arange(window_size), np.transpose(x_signal), deg=1, full=True
                    )

                    # Peng's formula - Convert to root-mean squared error, from squared error
                    # det_fluc = np.sqrt(np.mean(fluc / window_size))
                    # Richard's formula
                    det_fluc = np.mean(np.sqrt(fluc / window_size))
                    fluctuations[ch_idx, i_window_size] = det_fluc

            # get the positions of the first and last window sizes used for fitting
            fit_interval_first_window = np.argwhere(
                window_sizes >= fit_interval[0] * sampling_frequency
            )[0][0]
            fit_interval_last_window = np.argwhere(
                window_sizes <= fit_interval[1] * sampling_frequency
            )[-1][0]

            # take the previous to the first window size if the difference between the lower end of fitting and
            # the previous window is no more than 1% of the lower end of fitting and if the difference between the lower
            # end of fitting and the previous window is less than the difference between the lower end of fitting and the current first window
            if (
                np.abs(
                    window_sizes[fit_interval_first_window - 1] / sampling_frequency
                    - fit_interval[0]
                )
                <= fit_interval[0] / 100
            ):
                if np.abs(
                    window_sizes[fit_interval_first_window - 1] / sampling_frequency
                    - fit_interval[0]
                ) < np.abs(
                    window_sizes[fit_interval_first_window] / sampling_frequency
                    - fit_interval[0]
                ):
                    fit_interval_first_window = fit_interval_first_window - 1

            x = np.log10(
                window_sizes[fit_interval_first_window : fit_interval_last_window + 1]
            )
            y = np.log10(
                fluctuations[
                    ch_idx, fit_interval_first_window : fit_interval_last_window + 1
                ]
            )
            model = np.polyfit(x, y, 1)
            dfa_intercept[ch_idx] = model[1]
            dfa_array[ch_idx] = model[0]

    return (dfa_array, window_sizes, fluctuations, dfa_intercept)


def get_frequency_bins(frequency_range):
    """Get frequency bins for the frequency range of interest.

    Parameters
    ----------
    frequency_range : array, shape (1,2)
        The frequency range over which to create frequency bins.
        The lower edge should be equal or more than 1 Hz, and the upper edge should be equal or less than 150 Hz.

    Returns
    -------
    frequency_bins : list, shape (n_bins,2)
        The lower and upper range in Hz per frequency bin.
    """

    assert frequency_range[0] >= 1.0 and frequency_range[1] <= 150.0, (
        "The frequency range should cannot be less than 1 Hz or more than 150 Hz"
    )

    frequency_bin_delta = [1.0, 4.0]
    frequency_range_full = [frequency_bin_delta[1], 150]
    n_bins_full = 16

    # Create logarithmically-spaced bins over the full frequency range
    frequencies_full = np.logspace(
        np.log10(frequency_range_full[0]),
        np.log10(frequency_range_full[-1]),
        n_bins_full,
    )
    frequencies = np.append(frequency_bin_delta[0], frequencies_full)
    # Get frequencies that fall within the frequency range of interest
    myfrequencies = frequencies[
        np.where(
            (np.round(frequencies, 4) >= frequency_range[0])
            & (np.round(frequencies, 4) <= frequency_range[1])
        )[0]
    ]

    # Get all frequency bin ranges
    frequency_bins = [
        [myfrequencies[i], myfrequencies[i + 1]] for i in range(len(myfrequencies) - 1)
    ]

    return frequency_bins


def get_DFA_fitting_interval(frequency_interval):
    """Get a fitting interval for DFA computation.

    Parameters
    ----------
    frequency_interval : array, shape (1,2)
        The lower and upper bound of the frequency bin in Hz for which the fitting interval will be inferred.
        The fitting interval is where the regression line is fit for log-log coordinates of the fluctuation function vs. time windows.

    Returns
    -------
    fit_interval : array, shape (1,2)
        The lower and upper bound of the fitting range in seconds for a frequency bin.
    """

    # Upper fitting margin in seconds
    upper_fit = 30
    # Default lower fitting margins in seconds per frequency bin
    default_lower_fits = [
        5.0,
        5.0,
        5.0,
        3.981,
        3.162,
        2.238,
        1.412,
        1.122,
        0.794,
        0.562,
        0.398,
        0.281,
        0.141,
        0.1,
        0.1,
        0.1,
    ]

    frequency_bins = get_frequency_bins([1, 150])
    # Find the fitting interval. In case when frequency range is not exactly one from the defined frequency bins,
    # it finds the fitting interval of the bin for which the lowest of the provided frequencies falls into.
    idx_freq = np.where((np.array(frequency_bins)[:, 0] <= frequency_interval[0]))[0][
        -1
    ]

    fit_interval = [default_lower_fits[idx_freq], upper_fit]

    return fit_interval


def compute_spectrum_biomarkers(
    signal_matrix,
    sampling_frequency,
    spectrum_frequency_range,
    overlap=True,
    runtime="c",
    bad_idxes=[],
    biomarkers_to_compute=["fEI", "DFA"],
):
    """Compute spectral DFA and fEI for the frequencies provided.
    Parameters
    ----------
    signal_matrix : array, shape (n_channels,n_times)
        The signal to compute DFA and fEI for.
    biomarkers_to_compute: list[str]
        Contains the list of biomarkers to be computed. Options are DFA, fEI
        If fEI is mentioned, DFA is computed by default
    sampling_frequency : float
        The sample frequency in Hz.
    spectrum_frequency_range : array, shape (1,2)
        The frequency range over which to create frequency bins.
        The lower edge should be equal or more than 1 Hz, and the upper edge should be equal or less than 150 Hz.
    overlap : bool
        Whether 50% overlapping windows will be used.
        Default True
    method : str
        Whether to use Python or C-compiled version of DFA computation.
        Note that C-compiled version is 10 times faster. Default is C.
    bad_idxes : array, shape (1,)
        The indices of bad channels which will be ignored when DFA is computed. Will be NaNs.
        Default is empty list.

    Returns
    -------
    dfa_exponents_matrix : array, shape (n_channels,n_frequency_bins)
        Computed DFA exponents per frequency bin
    fei_values_matrix : array, shape (n_channels,n_frequency_bins)
        Computed fEI values per frequency bin
    """

    num_channels, num_timepoints = np.shape(signal_matrix)
    # Get frequency bins
    frequency_bins = get_frequency_bins(spectrum_frequency_range)

    output = {}
    if "DFA" or "fEI" in biomarkers_to_compute:
        output["DFA"] = np.zeros((num_channels, len(frequency_bins)))
    if "fEI" in biomarkers_to_compute:
        output["fEI"] = np.zeros((num_channels, len(frequency_bins)))

    for idx_frequency, frequency_bin in enumerate(frequency_bins):
        output_band = compute_band_biomarkers(
            signal_matrix,
            sampling_frequency,
            frequency_bin,
            overlap=overlap,
            runtime=runtime,
            bad_idxes=bad_idxes,
            biomarkers_to_compute=biomarkers_to_compute,
        )
        if "DFA" in biomarkers_to_compute or "fEI" in biomarkers_to_compute:
            output["DFA"][:, idx_frequency] = output_band["DFA"]

        if "fEI" in biomarkers_to_compute:
            output["fEI"][:, idx_frequency] = output_band["fEI"]

    return output


def compute_band_biomarkers(
    signal_matrix,
    sampling_frequency,
    frequency_range,
    overlap=True,
    runtime="c",
    bad_idxes=[],
    biomarkers_to_compute=["fEI", "DFA"],
):
    """Compute spectral DFA and fEI for a frequency band
    Parameters
    ----------
    signal_matrix : array, shape (n_channels,n_times)
        The signal to compute DFA and fEI for.
    biomarkers_to_compute: list[str]
        Contains the list of biomarkers to be computed. Options are DFA, fEI
        If fEI is mentioned, DFA is computed by default
    sampling_frequency : float
        The sample frequency in Hz.
    frequency_range : array, shape (1,2)
        The frequency range on which to compute the amplitude envelope and filter the data.
        The lower edge should be equal or more than 1 Hz, and the upper edge should be equal or less than 150 Hz.
    overlap : bool
        Whether 50% overlapping windows will be used.
        Default True
    method : str
        Whether to use Python or C-compiled version of DFA computation.
        Note that C-compiled version is 10 times faster. Default is C.
    bad_idxes : array, shape (1,)
        The indices of bad channels which will be ignored when DFA is computed. Will be NaNs.
        Default is empty list.

    Returns
    -------
    dfa_exponents_matrix : array, shape (n_channels,n_frequency_bins)
        Computed DFA exponents per frequency bin
    fei_values_matrix : array, shape (n_channels,n_frequency_bins)
        Computed fEI values per frequency bin
    """

    num_cores = multiprocessing.cpu_count()
    num_channels, num_timepoints = np.shape(signal_matrix)

    output = {}

    # Parameters
    fEI_window_seconds = 5
    fEI_overlap = 0.8

    # Get fit interval
    fit_interval = get_DFA_fitting_interval(frequency_range)
    DFA_compute_interval = fit_interval

    # Filter signal in the given frequency bin
    filtered_signal = mne.filter.filter_data(
        data=signal_matrix,
        sfreq=sampling_frequency,
        l_freq=frequency_range[0],
        h_freq=frequency_range[1],
        filter_length="auto",
        l_trans_bandwidth="auto",
        h_trans_bandwidth="auto",
        fir_window="hamming",
        phase="zero",
        fir_design="firwin",
        pad="reflect_limited",
        verbose=0,
    )

    filtered_signal = filtered_signal[
        :, 1 * sampling_frequency : filtered_signal.shape[1] - 1 * sampling_frequency
    ]
    # Compute amplitude envelope
    n_fft = next_fast_len(num_timepoints)
    amplitude_envelope = Parallel(n_jobs=num_cores, backend="threading", verbose=0)(
        delayed(hilbert)(filtered_signal[idx_channel, :], n_fft)
        for idx_channel in range(num_channels)
    )
    amplitude_envelope = np.abs(np.array(amplitude_envelope))

    if "DFA" in biomarkers_to_compute or "fEI" in biomarkers_to_compute:
        print(
            "Computing DFA for frequency range: %.2f - %.2f Hz"
            % (frequency_range[0], frequency_range[1])
        )
        (dfa_array, window_sizes, fluctuations, dfa_intercept) = DFA(
            amplitude_envelope,
            sampling_frequency,
            fit_interval,
            DFA_compute_interval,
            overlap,
            runtime,
            bad_idxes,
        )
        output["DFA"] = dfa_array

    if "fEI" in biomarkers_to_compute:
        print(
            "Computing fEI for frequency range: %.2f - %.2f Hz"
            % (frequency_range[0], frequency_range[1])
        )
        (fEI_outliers_removed, fEI_val, num_outliers, wAmp, wDNF) = fEI(
            amplitude_envelope,
            sampling_frequency,
            fEI_window_seconds,
            fEI_overlap,
            dfa_array,
            runtime,
            bad_idxes,
        )
        output["fEI"] = np.squeeze(fEI_outliers_removed)

    return output


def _create_window_indices(length_signal, length_window, window_offset):
    window_starts = np.arange(0, length_signal - length_window, window_offset)
    num_windows = len(window_starts)

    one_window_index = np.arange(0, length_window)
    all_window_index = repmat(one_window_index, num_windows, 1).astype(int)

    all_window_index = all_window_index + repmat(
        np.transpose(window_starts[np.newaxis, :]), 1, length_window
    ).astype(int)

    return all_window_index
