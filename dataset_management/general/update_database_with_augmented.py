import pickle
from augment_data.phenomenological_ses.make_phenomenological_ses import AugmentedSES
from database_definitions import make_db
from dataset_management.ultils.update_database import DerivedDoc, new_docs_from_computed
import numpy as np


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
    healthy_envelope_spectrum_mag = healthy_envelope_spectrum["mag"]
    healthy_envelope_spectrum_freq = healthy_envelope_spectrum["freq"]

    meta_data = pickle.loads(doc["meta_data"])
    fs = meta_data["sampling_frequency"]

    expected_fault_frequency = meta_data["derived"]["average_fault_frequency"]

    # TODO: Very important, Notice that setting the augmented amplitude using failure data is cheating: This is however now done for rapid iteration.
    # However, there will always be a disrepancy in practice.
    max_severity = db["processed"].distinct("severity")[-1]
    maximal_damage_for_mode = db["processed"].find_one({"envelope_spectrum": {"$exists": True},
                                                          "severity": max_severity,
                                                          "mode": doc["mode"]})
    damaged_envelope_spectrum = pickle.loads(maximal_damage_for_mode["envelope_spectrum"])["mag"]
    max_amplitude = np.max(damaged_envelope_spectrum)

    print("mode",doc["mode"],"max amp",max_amplitude)


    ases = AugmentedSES(healthy_ses=healthy_envelope_spectrum_mag,
                        healthy_ses_freq= healthy_envelope_spectrum_freq,
                        fs=fs,
                        fault_frequency=expected_fault_frequency,
                        peak_magnitude=max_amplitude,#0.01,
                        decay_percentage_over_interval=0.5
                        )  # TODO: Fix peak magnitude, providing augmentation parameters?

    envelope_spectrum = ases.get_augmented_ses()

    computed = [{"envelope_spectrum": pickle.dumps({"freq": ases.frequencies,
                                                "mag": envelope_spectrum}),
             "augmented": True,
             "augmentation_meta_data": {"this": "that"}}]

    new_docs = new_docs_from_computed(doc,computed)
    return new_docs


def main():
    from database_definitions import augmented

    augmented.delete_many({})

    # Compute augmented data
    query = {"envelope_spectrum": {"$exists": True}}
    DerivedDoc(query, "processed", "augmented", compute_augmentation_from_feature_doc).update_database(parallel=False)

    return augmented


if __name__ == "__main__":
    r = main()
