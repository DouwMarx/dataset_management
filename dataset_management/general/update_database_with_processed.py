import numpy as np
from signal_processing.spectra import env_spec, envelope
from signal_processing.filtering import bandpass

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
    signal = np.array(doc["time_series"])  
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
    envelope_spectrum = {"envelope_spectrum": {"freq": list(limit_frequency_components(freq)),
                                                            "mag": list(limit_frequency_components(mag)),
                                                            "phase": list(limit_frequency_components(phase))
                                                            },
                         "augmented": doc["augmented"]
                         }
    computed_features = [envelope_time_series, envelope_spectrum]

    new_docs = new_docs_from_computed(doc,computed_features)
    return new_docs



class ProcessData():
    def __init__(self, db_to_act_on,bandpass=False):
        self.db_to_act_on = db_to_act_on
        self.db, self.client = make_db(db_to_act_on)
        self.bandpass = bandpass

    def compute_features_from_time_series_doc(self, doc):
        meta_data = doc["meta_data"]
        fs = meta_data["sampling_frequency"]


        signal = np.array(doc["time_series"])

        if self.bandpass:
            bandwidth = 450
            center = 5600
            low = center - bandwidth/2
            high = center + bandwidth/2
            signal = bandpass(signal,low,high,fs)

        env = envelope(signal)
        freq, mag, phase = env_spec(signal, fs=fs)
        envelope_time_series = {"envelope_time_series": list(env),
                                "augmented": doc["augmented"]
                                }
        # Important: Notice that the DC gain is removed here
        envelope_spectrum = {"envelope_spectrum": {"freq": list(limit_frequency_components(freq)),
                                                   "mag": list(limit_frequency_components(mag)),
                                                   "phase": list(limit_frequency_components(phase))
                                                   },
                             "augmented": doc["augmented"]
                             }
        computed_features = [envelope_time_series, envelope_spectrum]

        new_docs = new_docs_from_computed(doc, computed_features)
        return new_docs

    # upper = 3750 + 312
    # lower = 3750 - 312



def main(db_to_act_on):
    db,client = make_db(db_to_act_on)
    db["processed"].delete_many({})

    if db_to_act_on in ["ims","ims_test"]:
        bandpass = True
    else:
        bandpass = False

    to_apply = ProcessData(db_to_act_on, bandpass=bandpass).compute_features_from_time_series_doc
    # Process the time data
    query = {"time_series": {"$exists": True}}
    DerivedDoc(query, "raw", "processed", to_apply,db_to_act_on).update_database(parallel=False)

    return db["processed"]


if __name__ == "__main__":
    r = main("phenomenological_rapid")
