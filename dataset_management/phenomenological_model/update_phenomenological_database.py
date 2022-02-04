import numpy as np
from augment_data.phenomenological_ses.make_phenomenological_ses import AugmentedSES
from sklearn.decomposition import PCA
from signal_processing.spectra import env_spec, envelope
import pickle
from database_definitions import client




def compute_signal_augmentation(entry):
    """
    Augments healthy data towards a faulty state for a given failure mode.

    Parameters
    ----------
    mode
    severity
    results_dict

    Returns
    -------

    """

    meta_data = pickle.loads(entry["meta_data"])
    fs = meta_data["sampling_frequency"]
    expected_fault_frequency = meta_data["derived"]["average_fault_frequency"]

    # Using all of the healthy ses as input means that the augmented dataset will have the noise of the training set
    # However, among the augmented dataset the "signal" will be identical
    # notice the "0" meaning that we are using healthy data
    healthy_ses = pickle.loads(entry["envelope_spectrum"])["mag"]  # [0] # Use the first one

    ases = AugmentedSES(healthy_ses=healthy_ses, fs=fs, fault_frequency=expected_fault_frequency,
                        peak_magnitude=0.03)  # TODO: Fix peak magnitude, providing augmentation parameters?
    envelope_spectrum = ases.get_augmented_ses()

    return {"augmented_envelope_spectrum": {"freq": ases.frequencies,
                                            "mag": envelope_spectrum}}


def add_processed_and_augmented_data(data_dict):
    """
    Adds features derived from the time domain signal as well as augmentation of the healthy data.
    This function is ran before models are trained and the encodings are added to dictionary.
    Parameters
    ----------
    data_dict

    Returns
    -------

    """
    for mode, dictionary in data_dict.items():
        for severity, measurements in dictionary.items():
            # Add the processed signal i.e. SES
            signal = measurements["time_domain"]
            fs = data_dict[mode][severity]["meta_data"]["sampling_frequency"]

            processed = compute_features_from_time_domain_signal(signal, fs)
            data_dict[mode][severity].update(processed)

            # Add the augmented signals
            augmented = compute_signal_augmentation(mode, severity, data_dict)
            data_dict[mode][severity].update(augmented)

    return data_dict


def limit_frequency_components(arr, fraction_of_spectrum_to_use=0.25):
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


def compute_encodings(data):
    """
    Train unsupervised models and compute the encodings that they result in.

    Parameters
    ----------
    data

    Returns
    -------

    """
    # Define models
    model_healthy_only = PCA(2)
    model_healthy_only.name = "healthy_only"
    model_healthy_and_augmented = PCA(2)
    model_healthy_and_augmented.name = "healthy_and_augmented"

    # Set up training data
    # Healthy data only
    all_healthy = [data[mode]["0"]["envelope_spectrum"]["mag"] for mode in list(data.keys())]
    healthy_train = limit_frequency_components(np.vstack(
        all_healthy))  # Healthy data from different "modes" even though modes don't technically exist when healthy

    # Healthy and augmented data
    all_augmented_modes = [data[mode]["1"]["augmented_envelope_spectrum"]["mag"] for mode in list(data.keys())]
    augmented_and_healthy_train = limit_frequency_components(np.vstack(all_healthy + all_augmented_modes))

    # Train the models
    model_healthy_only.fit(healthy_train)
    model_healthy_and_augmented.fit(augmented_and_healthy_train)

    # List of trained models
    models = [model_healthy_only, model_healthy_and_augmented]

    # Loop through all failure modes and severities.
    # For both the augmented and actual data, compute the expected encoding for each of the trained models.
    for mode_name, mode_data in data.items():
        for severity_name, severity_data in mode_data.items():
            data_type_dict = {}  # Data type refers to either real or augmented
            for data_type in ["envelope_spectrum", "augmented_envelope_spectrum"]:
                model_type_dict = {}
                for model in models:
                    encoding = model.transform(
                        limit_frequency_components(severity_data[data_type]["mag"]))

                    # Update the dictionary with the encodings
                    model_type_dict.update({model.name: encoding})
                data_type_dict.update({data_type + "_encoding": model_type_dict})
            data[mode_name][severity_name].update(data_type_dict)
    return data


def compute_augmentation_from_feature(doc, rapid = True):
    """
    Augments healthy data towards a faulty state for a given failure mode.

    Parameters
    ----------
    mode
    severity
    results_dict

    Returns
    -------

    """

    if rapid:
        db = client.phenomenological_rapid  # Use a specific dataset for rapid iteration

    else:
        db = client.phenomenological

    healthy_envelope_spectrum = db.failure_dataset.find_one({"envelope_spectrum": {"$exists":True},
                                                             "severity":"0",
                                                             "mode":doc["mode"]})#projection = "envelope_spectrum") # The value of the entry found to include
    # print(healthy_envelope_spectrum.keys())
    # print(pickle.loads(healthy_envelope_spectrum).keys())

    # healthy_envelope_spectrum = pickle.loads(healthy_envelope_spectrum["envelope_spectrum"])
    healthy_envelope_spectrum = pickle.loads(healthy_envelope_spectrum["envelope_spectrum"])
    healthy_envelope_spectrum =healthy_envelope_spectrum["mag"]

    print(healthy_envelope_spectrum.shape)

    meta_data = pickle.loads(doc["meta_data"])
    fs = meta_data["sampling_frequency"]

    expected_fault_frequency = meta_data["derived"]["average_fault_frequency"]

    # # Using all of the healthy ses as input means that the augmented dataset will have the noise of the training set
    # # However, among the augmented dataset the "signal" will be identical
    # # notice the "0" meaning that we are using healthy data
    # healthy_ses = pickle.loads(entry["envelope_spectrum"])["mag"]  # [0] # Use the first one

    ases = AugmentedSES(healthy_ses=healthy_envelope_spectrum, fs=fs, fault_frequency=expected_fault_frequency,
                        peak_magnitude=0.03)  # TODO: Fix peak magnitude, providing augmentation parameters?
    envelope_spectrum = ases.get_augmented_ses()

    return [{"envelope_spectrum": pickle.dumps({"freq": ases.frequencies,
                                               "mag": envelope_spectrum}),
            "augmented":"True",
            "augmentation_meta_data": {"this":"that"}}]


def compute_features_from_time_series_doc(doc,**kwargs):
    signal = pickle.loads(doc["time_series"])  # Get time signal
    # Get sampling frequency
    meta_data = pickle.loads(doc["meta_data"])
    fs = meta_data["sampling_frequency"]

    # Compute features # TODO: Make modular to select features to compute
    env = envelope(signal)  # The envelope of the signal can also be added to the computed features
    freq, mag, phase = env_spec(signal, fs=fs)

    envelope_time_series = {"envelope_time_series": pickle.dumps(env)}
    envelope_spectrum = {"envelope_spectrum": pickle.dumps({"freq": freq,
                                                            "mag": mag,
                                                            "phase": phase
                                                            })}
    computed_features = [envelope_time_series, envelope_spectrum]

    return computed_features


def new_derived_doc(query, function_to_apply, rapid=True):
    if rapid:
        db = client.phenomenological_rapid  # Use a specific dataset for rapid iteration

    else:
        db = client.phenomenological

    failure_dataset = db.failure_dataset

    # failure_dataset.delete_many({"augmented": {"$exists": True}})  # Remove if already existing

    # Loop through all the documents that satisfy the conditions of the query
    for doc in failure_dataset.find(query):
        # print(doc)
        computed = function_to_apply(doc, rapid = rapid) # TODO: Need keyword arguments to make this work. Or global variable?

        # Create a new document for each of the computed features, duplicate some of the original data
        for feature in computed:
            # TODO: Figure out how to deal with overwrites

            new_doc = {"mode": doc["mode"],
                       "severity": doc["severity"],
                       "meta_data": doc["meta_data"],
                       # feature_name: feature[feature_name]
                       }

            new_doc.update(feature)  # Add the newly computed data to a document containing the original meta data

            failure_dataset.insert_one(new_doc)
    return failure_dataset


def main():
    query = {"time_series": {"$exists": True}}
    new_derived_doc(query, compute_features_from_time_series_doc)

    query = {"envelope_spectrum": {"$exists": True}}
    fd = new_derived_doc(query, compute_augmentation_from_feature)
    return fd


if __name__ == "__main__":
    fd = main()
