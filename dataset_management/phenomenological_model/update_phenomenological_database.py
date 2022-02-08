import numpy as np
from augment_data.phenomenological_ses.make_phenomenological_ses import AugmentedSES
from sklearn.decomposition import PCA
from signal_processing.spectra import env_spec, envelope
import pickle
from database_definitions import raw, processed, augmented, encoding


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



def compute_augmentation_from_feature_doc(doc, rapid = True):
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

    healthy_envelope_spectrum = processed.find_one({"envelope_spectrum": {"$exists":True},
                                                             "severity":"0",
                                                             "mode":doc["mode"]})#projection = "envelope_spectrum") # The value of the entry found to include
    # print(healthy_envelope_spectrum.keys())
    # print(pickle.loads(healthy_envelope_spectrum).keys())

    # healthy_envelope_spectrum = pickle.loads(healthy_envelope_spectrum["envelope_spectrum"])
    healthy_envelope_spectrum = pickle.loads(healthy_envelope_spectrum["envelope_spectrum"])
    healthy_envelope_spectrum =healthy_envelope_spectrum["mag"]

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
            "augmented":True,
            "augmentation_meta_data": {"this":"that"}}]


def compute_features_from_time_series_doc(doc,**kwargs):
    signal = pickle.loads(doc["time_series"])  # Get time signal
    # Get sampling frequency
    meta_data = pickle.loads(doc["meta_data"])
    fs = meta_data["sampling_frequency"]

    # Compute features # TODO: Make modular to select features to compute
    env = envelope(signal)  # The envelope of the signal can also be added to the computed features
    freq, mag, phase = env_spec(signal, fs=fs)

    envelope_time_series = {"envelope_time_series": pickle.dumps(env),
                            "augmented":doc["augmented"]
                            }
    envelope_spectrum = {"envelope_spectrum": pickle.dumps({"freq": freq,
                                                            "mag": mag,
                                                            "phase": phase
                                                            }),
                         "augmented": doc["augmented"]
                         }
    computed_features = [envelope_time_series, envelope_spectrum]

    return computed_features

def compute_encoding_from_doc(doc,**kwargs):
    # TODO: make sure encodings are computed for both real and augmented data
    models = get_trained_models()
    encodings_for_models = []
    for model in models:
        encoding = model.transform(limit_frequency_components(pickle.loads(doc["envelope_spectrum"])["mag"]))
        encodings_for_models.append({"encoding": pickle.dumps(encoding),
                                     "model_used": model.name})
    return encodings_for_models


def new_derived_doc(query, source_collection, target_collection, function_to_apply, rapid=True):


    # Loop through all the documents that satisfy the conditions of the query
    for doc in source_collection.find(query):
        # print(doc)
        computed = function_to_apply(doc, rapid = rapid) # TODO: Need keyword arguments to make this work. Or global variable?

        # Create a new document for each of the computed features, duplicate some of the original data
        for feature in computed: # TODO: Could make use of insert_many?
            # TODO: Figure out how to deal with overwrites

            new_doc = {"mode": doc["mode"],
                       "severity": doc["severity"],
                       "meta_data": doc["meta_data"],
                       "augmented": doc["augmented"]
                       }

            new_doc.update(feature)  # Add the newly computed data to a document containing the original meta data

            target_collection.insert_one(new_doc)
    return target_collection


def get_trained_models():
    train_on_all_models = get_trained_models_train_on_all()
    trained_on_mode_models = get_trained_on_specific_failure_mode()

    return train_on_all_models + trained_on_mode_models

def get_trained_models_train_on_all():
    # Define models
    model_healthy_only = PCA(2)
    model_healthy_only.name = "healthy_only_pca"
    model_healthy_and_augmented = PCA(2)
    model_healthy_and_augmented.name = "healthy_and_augmented_pca"

    # Set up training data
    # Healthy data only
    all_healthy = [pickle.loads(doc["envelope_spectrum"])["mag"] for doc in processed.find({"envelope_spectrum": {"$exists": True},
                                                                                            "augmented":False,
                                                                                            "severity":"0"})]
    healthy_train = limit_frequency_components(np.vstack(
        all_healthy))  # Healthy data from different "modes" even though modes don't technically exist when healthy

    # # Healthy and augmented data
    all_augmented_modes = [pickle.loads(doc["envelope_spectrum"])["mag"] for doc in augmented.find({"envelope_spectrum": {"$exists": True},
                                                                                                    "severity":"9",
                                                                                                    "augmented":True})]
    augmented_and_healthy_train = limit_frequency_components(np.vstack(all_healthy + all_augmented_modes))

    # Train the models
    model_healthy_only.fit(healthy_train)
    model_healthy_and_augmented.fit(augmented_and_healthy_train)

    # List of trained models
    models = [model_healthy_only, model_healthy_and_augmented]
    return models


def get_trained_on_specific_failure_mode():
    # Define models
    failure_modes = augmented.distinct("mode")

    # Create a model for each failure mode
    models = [PCA(2) for failure_mode in failure_modes]

    # Give each model a name
    trained_models = []
    for model,mode_name in zip(models, failure_modes):
        model.name = "PCA2_health_and_" + mode_name

        # Set up training data
        # Healthy data only
        healthy = [pickle.loads(doc["envelope_spectrum"])["mag"] for doc in processed.find({"envelope_spectrum": {"$exists": True},
                                                                                                "augmented":False,
                                                                                                "severity":"0",
                                                                                                "mode": mode_name
                                                                                            })]
        healthy_train = limit_frequency_components(np.vstack(healthy)) # Using all of the healthy data from all "modes" (even though healthy

        # Augmented data
        all_augmented_modes = [pickle.loads(doc["envelope_spectrum"])["mag"] for doc in augmented.find({"envelope_spectrum": {"$exists": True},
                                                                                                        "severity":"9", # Using the maximum severity only during training
                                                                                                        "mode": mode_name,
                                                                                                        "augmented":True})]

        all_augmented_modes = limit_frequency_components(np.vstack(all_augmented_modes)) # Using all of the healthy data from all "modes" (even though healthy
        # print(all_augmented_modes[0].shape)
        augmented_and_healthy_train = np.vstack([healthy_train,all_augmented_modes])

        # # Train the models
        model.fit(augmented_and_healthy_train)
        trained_models.append(model)
    return models


# def main():
processed.delete_many({})
augmented.delete_many({})
encoding.delete_many({})


# Process the time data
query = {"time_series": {"$exists": True}}
new_derived_doc(query,raw,processed, compute_features_from_time_series_doc)

# Compute augmented data
query = {"envelope_spectrum": {"$exists": True}}
new_derived_doc(query, processed, augmented, compute_augmentation_from_feature_doc)

# Apply encoding for both augmented and not augmented data
query = {"augmented": True}
new_derived_doc(query, augmented, encoding, compute_encoding_from_doc)

query = {"augmented": False,"envelope_spectrum":{"$exists":True}}
new_derived_doc(query, processed, encoding, compute_encoding_from_doc)

def main():
    return

if __name__ == "__main__":
    fd = main()
