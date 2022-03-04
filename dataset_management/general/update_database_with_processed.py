import numpy as np
from signal_processing.spectra import env_spec, envelope
import pickle

from database_definitions import make_db
from dataset_management.ultils.update_database import DerivedDoc,new_docs_from_computed


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
    siglen = len(arr)
    use = int(siglen * fraction_of_spectrum_to_use)
    # return arr[:, 1:use]
    return arr[1:use]


def compute_features_from_time_series_doc(doc):
    signal = np.array(doc["time_series"])  # Get time signal
    # print(signal)
    # Get sampling frequency
    meta_data = doc["meta_data"]
    fs = meta_data["sampling_frequency"]

    # Compute features # TODO: Make modular to select features to compute
    env = envelope(signal)  # The envelope of the signal can also be added to the computed features
    freq, mag, phase = env_spec(signal, fs=fs)

    envelope_time_series = {"envelope_time_series": list(env),
                            "augmented": doc["augmented"]
                            }

    # print(freq.shape)
    # print(phase.shape)
    # print(mag.shape)
    # print("in dta", limit_frequency_components(freq.reshape(1, -1)).flatten().shape)

    # Important: Notice that the DC gain is removed here
    envelope_spectrum = {"envelope_spectrum": pickle.dumps({"freq": limit_frequency_components(freq),
                                                            "mag": limit_frequency_components(mag),
                                                            "phase": limit_frequency_components(phase)
                                                            }),
                         "augmented": doc["augmented"]
                         }
    computed_features = [envelope_time_series, envelope_spectrum]

    new_docs = new_docs_from_computed(doc,computed_features)
    return new_docs


def main():
    db_to_act_on = "phenomenological_rapid"
    db,client = make_db(db_to_act_on)
    db["processed"].delete_many({})

    # Process the time data
    query = {"time_series": {"$exists": True}}
    DerivedDoc(query, "raw", "processed", compute_features_from_time_series_doc,db_to_act_on).update_database(parallel=True)

    return db["processed"]


if __name__ == "__main__":
    r = main()
