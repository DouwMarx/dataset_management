import numpy as np
import scipy.stats
from pymongo import MongoClient
from joblib import Parallel, delayed
from scipy.stats import entropy
from tqdm import tqdm
from database_definitions import make_db


# Most of the features from here
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6823731

def get_rms(sig):
    return np.sqrt(np.mean(np.square(sig - np.mean(sig))))


def get_sra(sig):
    # Average of the square root absolute value of the signal
    return np.mean(np.sqrt(np.abs(sig - np.mean(sig))))


def get_kurtosis(sig):
    return scipy.stats.kurtosis(sig)


def get_crest_factor(sig):
    return np.max(np.abs(sig)) / get_rms(sig)


def get_entropy(sig):
    return np.nan_to_num(scipy.stats.differential_entropy(sig))


def get_skewness(sig):
    return scipy.stats.skew(sig)


def get_frequency_features(sig, rpm, fs, faults_per_revolution_for_each_mode):
    rotation_rate = rpm / 60

    fault_freqs_per_mode = {mode: freq * rotation_rate for mode, freq in faults_per_revolution_for_each_mode.items()}

    # # Square signal to get envelope and remove dc component
    # sig = np.array(sig)**2

    # Get the envelope by filtering and then taking the hilbert transform
    b, a = scipy.signal.butter(4, [fs / 4, fs * (3 / 8)], fs=fs, btype="bandpass")
    sig = scipy.signal.filtfilt(b, a, sig)
    sig = np.abs(scipy.signal.hilbert(sig))

    sig = sig - np.mean(sig)  # Remove dc component

    # Compute the FFT of the envelope
    fft = np.fft.fft(sig)  # Square the signal to get the envelope
    fft = np.abs(fft) / len(fft)  # Use fft magnitude and Normalize the fft
    freqs = np.fft.fftfreq(len(sig), 1 / fs)

    # Only use the one-sided spectrum and discard the dc component
    fft = fft[:len(fft) // 2][1:]
    freqs = freqs[:len(freqs) // 2][1:]

    # fft = fft**2# Using squared spectrum

    frequency_features = {"fft": list(fft),
                          "spectral_entropy": np.nan_to_num(
                              scipy.stats.differential_entropy(fft + 0.00001))}  # Store the fft for later use

    # Find the index of the frequency that is closest to the respective fault frequencies
    for mode, expected_freq in fault_freqs_per_mode.items():
        if expected_freq > fs / 2:
            raise Warning("Expected fault frequency above the Nyquist frequency")
        for harmonic in range(1, 6):
            index = np.argmin(np.abs(
                freqs - expected_freq * harmonic))  # The index of the frequency that is closest to the expected frequency
            frequency_features[mode + "_h" + str(harmonic)] = np.mean(
                fft[index - 5:index + 5])  # The value of the fft at that index
    return frequency_features


# Store all functions used to compute the features in a dictionary
feature_dict = {"rms": get_rms,
                "sra": get_sra,
                "kurtosis": get_kurtosis,
                "crest_factor": get_crest_factor,
                "entropy": get_entropy,
                "skewness": get_skewness,
                "frequency_features": get_frequency_features
                }


def process(doc,dataset_name):
    client = MongoClient()
    db = client[dataset_name]
    collection = db["raw"]

    faults_per_revolution = db["meta_data"].find_one({"_id": "meta_data"})["n_faults_per_revolution"]

    time_series = doc["time_series"]

    # Loop though each entry in the collection
    for key, function in feature_dict.items():

        if key == "frequency_features":
            freq_features = function(time_series, doc["rpm"], doc["sampling_frequency"],faults_per_revolution)
            for freq_key, freq_value in freq_features.items():
                collection.update_one({"_id": doc["_id"]}, {"$set": {freq_key: freq_value}})
        else:
            collection.update_one({"_id": doc["_id"]}, {"$set": {key: function(time_series)}})
    client.close()


# Add the data to the database in parallel
dataset_to_use = "lms"

client = MongoClient()
db = client[dataset_to_use]

# Run in parallel
Parallel(n_jobs=10)(delayed(process)(doc,dataset_to_use) for doc in tqdm(db["raw"].find()))
