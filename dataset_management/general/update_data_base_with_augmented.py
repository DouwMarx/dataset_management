import pickle
from augment_data.phenomenological_ses.make_phenomenological_ses import AugmentedSES
from dataset_management.ultils.update_database import new_derived_doc
from database_definitions import make_db


def compute_augmentation_from_feature_doc(doc):
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

    # Instantiate a own database
    db, client = make_db() # Depending on this input kwarg?, a different dataset could be selected.

    # TODO: The augmentation is currently envelope spectrum specific

    healthy_envelope_spectrum = db["processed"].find_one({"envelope_spectrum": {"$exists": True},
                                                          "severity": "0",
                                                          "mode": doc[
                                                              "mode"]})

    healthy_envelope_spectrum = pickle.loads(healthy_envelope_spectrum["envelope_spectrum"])
    healthy_envelope_spectrum = healthy_envelope_spectrum["mag"]

    meta_data = pickle.loads(doc["meta_data"])
    fs = meta_data["sampling_frequency"]

    expected_fault_frequency = meta_data["derived"]["average_fault_frequency"]

    # # Using all of the healthy ses as input means that the augmented dataset will have the noise of the training set
    # # However, among the augmented dataset the "signal" will be identical
    # # notice the "0" meaning that we are using healthy data
    # healthy_ses = pickle.loads(entry["envelope_spectrum"])["mag"]  # [0] # Use the first one

    ases = AugmentedSES(healthy_ses=healthy_envelope_spectrum, fs=fs, fault_frequency=expected_fault_frequency,
                        peak_magnitude=0.01,
                        percentage_of_freqs_to_decay_99_percent=0.5)  # TODO: Fix peak magnitude, providing augmentation parameters?

    envelope_spectrum = ases.get_augmented_ses()

    return [{"envelope_spectrum": pickle.dumps({"freq": ases.frequencies,
                                                "mag": envelope_spectrum}),
             "augmented": True,
             "augmentation_meta_data": {"this": "that"}}]


def main():
    from database_definitions import augmented

    augmented.delete_many({})

    # Compute augmented data
    query = {"envelope_spectrum": {"$exists": True}}
    new_derived_doc(query, "processed", "augmented", compute_augmentation_from_feature_doc)

    return augmented


if __name__ == "__main__":
    r = main()
