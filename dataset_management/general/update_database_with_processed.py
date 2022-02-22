from signal_processing.spectra import env_spec, envelope
import pickle
from dataset_management.ultils.update_database import new_derived_doc


def limit_frequency_components(arr, fraction_of_spectrum_to_use=0.1):
    """
    Removes the DC component of the Squared envelope spectrum.
    Furthermore, it uses only a fraction of the total spectrum
    Parameters
    ----------
    fraction_of_spectrum_to_use
    arr

    Returns
    -------

    """
    siglen = arr.shape[1]
    use = int(siglen * fraction_of_spectrum_to_use)
    return arr[:, 1:use]


def compute_features_from_time_series_doc(doc):
    signal = pickle.loads(doc["time_series"])  # Get time signal
    # Get sampling frequency
    meta_data = pickle.loads(doc["meta_data"])
    fs = meta_data["sampling_frequency"]

    # Compute features # TODO: Make modular to select features to compute
    env = envelope(signal)  # The envelope of the signal can also be added to the computed features
    freq, mag, phase = env_spec(signal, fs=fs)

    envelope_time_series = {"envelope_time_series": pickle.dumps(env),
                            "augmented": doc["augmented"]
                            }
    envelope_spectrum = {"envelope_spectrum": pickle.dumps({"freq": freq,
                                                            "mag": mag,
                                                            "phase": phase
                                                            }),
                         "augmented": doc["augmented"]
                         }
    computed_features = [envelope_time_series, envelope_spectrum]
    print("now")

    return computed_features


def main():
    from database_definitions import processed

    processed.delete_many({})

    # Process the time data
    query = {"time_series": {"$exists": True}}
    new_derived_doc(query, "raw", "processed", compute_features_from_time_series_doc)

    return processed


if __name__ == "__main__":
    r = main()
