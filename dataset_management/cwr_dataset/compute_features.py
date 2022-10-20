import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import entropy
from database_definitions import make_db

oc = "oc2"
db, client = make_db("cwr_" + oc)

# Most of the features from here
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6823731

def get_rms(sig):
    return np.sqrt(np.mean(np.square(sig)))

def get_sra(sig):
    # Average of the square root absolute value of the signal
    return np.mean(np.sqrt(np.abs(sig)))

def get_kurtosis(sig):
    return np.mean(np.power(sig, 4)) / np.power(np.mean(np.power(sig, 2)), 2)

def get_crest_factor(sig):
    return np.max(np.abs(sig)) / get_rms(sig)

def get_entropy(sig):
    # min-max normalization
    sig = (sig - np.min(sig)) / (np.max(sig) - np.min(sig))
    return entropy(sig+1e-10) # Make sure there are no zero values in the signal

def get_skewness(sig):
    return np.mean(np.power(sig, 3)) / np.power(np.mean(np.power(sig, 2)), 3 / 2)

def get_frequency_features(sig, rpm=1,fs=1):
    rotation_rate = rpm / 60
    expected_fault_frequencies = {"ball": 2.357 * rotation_rate,
                                  "outer": 3.585 * rotation_rate,
                                  "inner": 5.415 * rotation_rate}

    # Compute the FFT
    fft = np.fft.fft(sig)
    fft = np.abs(fft)
    freqs = np.fft.fftfreq(len(sig), 1 / fs)

    # Only use the one-sided spectrum
    fft = fft[:len(fft) // 2]
    freqs = freqs[:len(freqs) // 2]


    # Find the index of the frequency that is closest to the respective fault frequencies
    frequency_features = {}
    for mode, expected_freq in expected_fault_frequencies.items():
        if expected_freq>fs/2:
            raise Warning("Expected fault frequency above the Nyquist frequency")
        for harmonic in range(2,6):
            index = np.argmin(np.abs(freqs - expected_freq*harmonic))
            frequency_features[mode + "_h" + str(harmonic)] = fft[index]

    return frequency_features

feature_dict = {"rms": get_rms,
                "sra": get_sra,
                "kurtosis": get_kurtosis,
                "crest_factor": get_crest_factor,
                "entropy": get_entropy,
                "skewness": get_skewness,
                "frequency_features": get_frequency_features
                }


collection = db["raw"]


# for example in list(collection.find({"severity":0.53}))[:10]:
#     scores = get_frequency_features(example["time_series"], example["rpm"], example["sampling_frequency"])
#     mode = example["mode"]
#     print("True mode:", mode, "highest score:", max(scores, key=scores.get))


# Loop though each entry in the collection
for doc in collection.find():
    for key, function in feature_dict.items():
        if key == "frequency_features":
            freq_features = function(doc["time_series"], rpm=doc["rpm"], fs=doc["sampling_frequency"])
            for freq_key, freq_value in freq_features.items():
                collection.update_one({"_id": doc["_id"]}, {"$set": {freq_key: freq_value}})
        else:
            collection.update_one({"_id": doc["_id"]}, {"$set": {key: function(doc["time_series"])}})
    print(doc.keys())
