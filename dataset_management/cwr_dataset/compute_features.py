import matplotlib.pyplot as plt
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
    return np.sqrt(np.mean(np.square(sig-np.mean(sig))))

def get_sra(sig):
    # Average of the square root absolute value of the signal
    return np.mean(np.sqrt(np.abs(sig-np.mean(sig))))

def get_log_kurtosis(sig):
    return scipy.stats.kurtosis(sig)

def get_crest_factor(sig):
    return np.max(np.abs(sig)) / get_rms(sig)

def get_entropy(sig):
    return np.nan_to_num(scipy.stats.differential_entropy(sig))

def get_skewness(sig):
    return scipy.stats.skew(sig)

def get_frequency_features(sig, rpm=1,fs=1):
    rotation_rate = rpm / 60
    expected_fault_frequencies = {"ball": 2.357 * rotation_rate,
                                  "outer": 3.585 * rotation_rate,
                                  "inner": 5.415 * rotation_rate}

    # Compute the FFT of the envelope
    fft = np.fft.fft(np.array(sig)**2) # Square the signal to get the envelope
    fft = np.abs(fft)/len(fft) # Use magnitude and Normalize the fft
    freqs = np.fft.fftfreq(len(sig), 1 / fs)

    # Only use the one-sided spectrum
    fft = fft[:len(fft) // 2]
    freqs = freqs[:len(freqs) // 2]

    # fft = fft**2# Using squared spectrum

    frequency_features = {"fft": list(fft),
                          "spectral_entropy":np.nan_to_num(scipy.stats.differential_entropy(fft))} # Store the fft for later use


    # Find the index of the frequency that is closest to the respective fault frequencies
    for mode, expected_freq in expected_fault_frequencies.items():
        if expected_freq>fs/2:
            raise Warning("Expected fault frequency above the Nyquist frequency")
        for harmonic in range(1,6):
            index = np.argmin(np.abs(freqs - expected_freq*harmonic)) # The index of the frequency that is closest to the expected frequency
            frequency_features[mode + "_h" + str(harmonic)] = np.mean(fft[index-5:index+5]) # The value of the fft at that index
    return frequency_features

# Store all functions used to compute the features in a dictionary
feature_dict = {"rms": get_rms,
                "sra": get_sra,
                "kurtosis": get_log_kurtosis,
                "crest_factor": get_crest_factor,
                "entropy": get_entropy,
                "skewness": get_skewness,
                "frequency_features": get_frequency_features
                }

def process(doc):
    client = MongoClient()
    db = client["cwr"]
    collection = db["raw"]

    time_series = doc["time_series"]

    # Loop though each entry in the collection
    for key, function in feature_dict.items():

        if key == "frequency_features":
            freq_features = function(time_series, rpm=doc["rpm"], fs=doc["sampling_frequency"])
            for freq_key, freq_value in freq_features.items():
                collection.update_one({"_id": doc["_id"]}, {"$set": {freq_key: freq_value}})
        else:
            collection.update_one({"_id": doc["_id"]}, {"$set": {key: function(time_series)}})
    client.close()

# Add the data to the database in parallel

client = MongoClient()
db = client["cwr"]

Parallel(n_jobs=10)(delayed(process)(doc) for doc in tqdm(db["raw"].find()))
