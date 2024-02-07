# Load the dataframe containing all signals and meta-data
import pathlib

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.signal import butter, lfilter
from dataset_management.cwr_dataset.write_data_to_standard_structure import get_cwru_data_frame
from dataset_management.ultils.gradient_prescription import TriangularPeaks


def bandpass(signals, lower, upper, fs, order=6):
    b, a = butter(order, [lower / (0.5 * fs), upper / (0.5 * fs)], btype='band')
    return filtfilt(b, a, signals, axis=-1)

def get_envelope_spectrum(signals, fs, filter_order=50, low_cut_as_fraction_of_fs=1 / 4, high_cut_as_fraction_of_fs=3 / 8, truncate_at_fraction_of_nyquist=1 / 2):
    """
    Get the envelope spectrum of a signal
    """

    # Squeeze the channel dimesion if it is 1
    signals = np.squeeze(signals)

    # Detrend the signal over the last axis (time)
    signals = signals - np.mean(signals, axis=-1, keepdims=True)

    # Filter the signal
    lowcut = low_cut_as_fraction_of_fs * fs
    highcut = high_cut_as_fraction_of_fs * fs

    filtered_signals = bandpass(signals, lowcut, highcut, fs, order=filter_order)

    analytic_signal = hilbert(filtered_signals) # Get the analytic signal
    amplitude_envelope = np.abs(analytic_signal) # Get the amplitude envelope

    # Detrend the envelope
    amplitude_envelope = amplitude_envelope - np.mean(amplitude_envelope, axis=-1, keepdims=True)


    envelope_spectrum = np.abs(np.fft.rfft(amplitude_envelope,axis=-1)) # Get the envelope spectrum
    freqs = np.fft.rfftfreq(amplitude_envelope.shape[-1], d=1/fs) # Get the frequencies for the spectrum

    if truncate_at_fraction_of_nyquist is not None:
        truncate_at = int(truncate_at_fraction_of_nyquist * envelope_spectrum.shape[-1])
        envelope_spectrum = envelope_spectrum[:,:truncate_at]
        freqs = freqs[:truncate_at]


    return envelope_spectrum, freqs


def get_derived_features_and_domain_knowledge(row):
    """
    Get the derived features and domain knowledge from the row
    """
    signals = row["Signals"]
    fault_frequency = row["Fault Frequency"]
    fs = float(row["Sampling Rate [kHz]"]) * 1000

    envelope_spectrum, freqs = get_envelope_spectrum(signals, fs=fs, filter_order=20)
    peak_simulator = TriangularPeaks(freqs_to_simulate_for=freqs)
    expected_fault_spectrum = peak_simulator.get_expected_fault_behaviour(1, fault_frequency)
    return {"Envelope Spectrum": envelope_spectrum, "Spectrum Freqs": freqs, "Expected Fault Spectrum": expected_fault_spectrum}


if __name__ == "__main__":
    this_folder = pathlib.Path(__file__).parent
    df = get_cwru_data_frame(min_average_events_per_rev=50,
                             path_to_write= this_folder.joinpath("cwru_env_spec.pkl"),
                             data_path=this_folder
                             )
    # Further limit to DE measurement location (Measure at the location where the faulty is present)

    new_columns = df.apply(get_derived_features_and_domain_knowledge, axis=1, result_type="expand")
    df[new_columns.columns] = new_columns
    # Write to pickle
    df.to_pickle("envelope_spectrum_and_expected_fault_spectrum.pkl")

    df = df[df["Measurement Location"] == "DE"]
    for random_row_number in np.random.choice(df.index, 20):
        row = df.loc[random_row_number]
        fault_frequency = row["Fault Frequency"]

        # derived_features_and_knowledge = get_derived_features_and_domain_knowledge(row)
        envelope_spectrum = row["Envelope Spectrum"]
        freqs = row["Spectrum Freqs"]
        prescription = row["Expected Fault Spectrum"]

        info = row.drop(["Signals", "Envelope Spectrum", "Spectrum Freqs", "Expected Fault Spectrum"])
        info = info.to_dict()
        info = "".join(["{}: {}\n".format(key, value) for key, value in info.items()])
        plt.figure()
        # plt.title("Envelope spectrum for fault mode: {}".format( info))
        # Add the info as text on the background
        plt.text(0.5, 0.5, info, fontsize=12, ha='left', va='top', alpha=0.5, transform=plt.gca().transAxes)


        plt.plot(freqs, envelope_spectrum[0], label="True")
        plt.plot(freqs, prescription*max(envelope_spectrum[0]), label="Prescribed")

        #Plot faulty frequency as vertical line
        plt.axvline(fault_frequency, color='r', linestyle='--', label="Fault Frequency")

        plt.show()

