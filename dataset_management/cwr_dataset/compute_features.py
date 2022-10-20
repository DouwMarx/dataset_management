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
    return entropy(sig/np.max(np.abs(sig)))

def get_skewness(sig):
    return np.mean(np.power(sig, 3)) / np.power(np.mean(np.power(sig, 2)), 3 / 2)

def get_frequency_features(sig, meta_data):
    fs = meta_data["sampling_frequency"]
    expected_fault_frequency = meta_data["expected_fault_frequency"]

    # Compute the FFT
    fft = np.fft.fft(sig)
    fft = np.abs(fft)
    freqs = np.fft.fftfreq(len(sig), 1 / fs)

    # Only use the one sided spectrum
    fft = fft[:len(fft) // 2]
    freqs = freqs[:len(freqs) // 2]

    # Find the index of the frequency that is closest to the expected fault frequency
    index = np.argmin(np.abs(freqs - expected_fault_frequency))

    # Return the amplitude of the frequency closest to the expected fault frequency
    return fft[index]




feature_dict = {"rms": get_rms,
                "sra": get_sra,
                "kurtosis": get_kurtosis,
                "crest_factor": get_crest_factor,
                "entropy": get_entropy,
                "skewness": get_skewness,
                }

collection = db["raw"]

example = collection.find_one()["time_series"]
example_meta_data = {"sampling_frequency": 1000, "expected_fault_frequency": 10}

a = get_frequency_features(example, example_meta_data)

# # Loop though each entry in the collection
# for doc in collection.find():
#     for key, function in feature_dict.items():
#         collection.update({"_id": doc["_id"]}, {"$set": {key: function(doc["time_series"])}})
#     print(doc.keys())
