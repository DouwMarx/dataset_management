import numpy as np
import scipy.stats
from pymongo import MongoClient
from joblib import Parallel, delayed
from scipy.stats import entropy
from tqdm import tqdm
from database_definitions import make_db
np.seterr(all='raise')
import warnings
warnings.simplefilter('error')
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


    # # Square signal to get rudimentary envelope and remove dc component
    # sig = np.array(sig -np.mean(sig))**2

    # Get the envelope by filtering and then taking the hilbert transform
    b, a = scipy.signal.butter(4, [fs / 4, fs * (3 / 8)], fs=fs, btype="bandpass")
    sig = scipy.signal.filtfilt(b, a, sig)
    sig = np.abs(scipy.signal.hilbert(sig))

    sig = sig - np.mean(sig)  # Remove dc component

    # Compute the FFT of the envelope
    fft = np.fft.fft(sig)  # Square the signal to get the envelope
    fft = np.abs(fft) / len(fft)  # Use fft magnitude and Normalize the fft
    freqs = np.fft.fftfreq(len(sig), 1 / fs)

    # Only use the one-sided spectrum and discard the dc component, further use only half of the positive frequencies (1/4 of the Nyquist frequency)
    fft = fft[:len(fft) // 4][1:]
    freqs = freqs[:len(freqs) // 4][1:]

    spectral_entropy = np.nan_to_num(scipy.stats.differential_entropy(fft + 0.00001))

    frequency_features = {"fft": list(fft),
                          "spectral_entropy": spectral_entropy}  # Store the fft for later use

    fault_freqs_per_mode = {mode: faults_per_rev * rotation_rate for mode, faults_per_rev in faults_per_revolution_for_each_mode.items() if mode != "healthy"}
    # Find the index of the frequency that is closest to the respective fault frequencies
    for mode, expected_freq in fault_freqs_per_mode.items():
        if expected_freq > fs / 2:
            raise Warning("Expected fault frequency above the Nyquist frequency")

        for harmonic in range(2, 6):
            if mode == "healthy":
                frequency_features[mode + "_h" + str(harmonic)] = None
            else:
                index = np.argmin(np.abs(
                    freqs - expected_freq * harmonic))  # The index of the frequency that is closest to the expected frequency

                n_band = 2
                if index < n_band or index > len(freqs) - n_band:
                    raise Warning("Index out of range for  " + mode + " harmonic " + str(harmonic), "rpm: " + str(rpm),)

                frequency_features[mode + "_h" + str(harmonic)] = np.mean(fft[index - n_band:index + n_band])


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
    np.seterr(all='raise')
    import warnings
    warnings.simplefilter('error')

    client = MongoClient()
    db = client[dataset_name]
    collection = db["raw"]

    db_meta_data = db["meta_data"].find_one({"_id": "meta_data"})
    faults_per_revolution = db_meta_data["n_faults_per_revolution"]
    sampling_frequency = db_meta_data["sampling_frequency"]

    time_series = doc["time_series"]

    # Loop though each entry in the collection
    for key, function in feature_dict.items():

        if key == "frequency_features":
        # Catch any runtime warnings
            try:
                freq_features = function(time_series, doc["rpm"], sampling_frequency,faults_per_revolution)
                for freq_key, freq_value in freq_features.items():
                    collection.update_one({"_id": doc["_id"]}, {"$set": {freq_key: freq_value}})
            except RuntimeWarning as e:
                doc["time_series"] = ""
                doc["fft"] = ""
                print("RuntimeWarning for doc: " + str(doc))
                print(e)
        else:
            collection.update_one({"_id": doc["_id"]}, {"$set": {key: function(time_series)}})
    client.close()


def main(dataset_to_use):
    # Add the data to the database in parallel


    # import warnings
    # np.seterr(all='warn')
    # warnings.filterwarnings('error')

    client = MongoClient()
    db = client[dataset_to_use]

    # Run in parallel
    Parallel(n_jobs=10)(delayed(process)(doc,dataset_to_use) for doc in tqdm(db["raw"].find()))

    # # Run in series
    # for doc in tqdm(db["raw"].find()):
    #     process(doc,dataset_to_use)

    client.close()

if __name__ == "__main__":
    dataset_to_use = "lms"
    main(dataset_to_use)