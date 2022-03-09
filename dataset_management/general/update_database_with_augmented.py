import pickle
from augment_data.phenomenological_ses.make_phenomenological_ses import AugmentedSES
from database_definitions import make_db
from dataset_management.ultils.update_database import DerivedDoc, new_docs_from_computed
import numpy as np


class Augmentation():
    def __init__(self, db_to_act_on):
        self.db_to_act_on = db_to_act_on
        # Instantiate a own database
        self.db, self.client = make_db(
            db_to_act_on)  # Depending on this input kwarg?, a different dataset could be selected.
        self.max_severity = self.db["processed"].distinct("severity")[-1]

        self.failure_modes = ["ball", "inner", "outer"]

        # TODO: Very important, Notice that setting the augmented amplitude using failure data is cheating: This is however now done for rapid iteration.
        # However, there will always be a disrepancy in practice.

        # for mode in "ball"
        # maximal_damage_for_mode = self.db["processed"].find_one({"envelope_spectrum": {"$exists": True},
        #                                                          "severity": self.max_severity,
        #                                                          "mode": doc["mode"]})
        # damaged_envelope_spectrum = pickle.loads(maximal_damage_for_mode["envelope_spectrum"])["mag"]
        # max_amplitude = np.max(damaged_envelope_spectrum)

    def compute_augmentation_from_feature_doc(self, doc):

        # TODO: The augmentation is currently envelope spectrum specific

        healthy_envelope_spectrum = self.db["processed"].find_one({"envelope_spectrum": {"$exists": True},
                                                                   "severity": "0",
                                                                   "mode": doc[
                                                                       "mode"]})

        healthy_envelope_spectrum = pickle.loads(healthy_envelope_spectrum["envelope_spectrum"])
        healthy_envelope_spectrum_mag = healthy_envelope_spectrum["mag"]
        healthy_envelope_spectrum_freq = healthy_envelope_spectrum["freq"]

        # meta_data = pickle.loads(doc["meta_data"])
        meta_data = doc["meta_data"]
        fs = meta_data["sampling_frequency"]

        expected_fault_frequency = meta_data["derived"]["average_fault_frequency"]

        # TODO: Very important, Notice that setting the augmented amplitude using failure data is cheating: This is however now done for rapid iteration.
        # However, there will always be a disrepancy in practice.
        max_severity = self.db["processed"].distinct("severity")[-1]
        maximal_damage_for_mode = self.db["processed"].find_one({"envelope_spectrum": {"$exists": True},
                                                                 "severity": max_severity,
                                                                 "mode": doc["mode"]})
        damaged_envelope_spectrum = pickle.loads(maximal_damage_for_mode["envelope_spectrum"])["mag"]
        max_amplitude = np.max(damaged_envelope_spectrum)

        ases = AugmentedSES(healthy_ses=healthy_envelope_spectrum_mag,
                            healthy_ses_freq=healthy_envelope_spectrum_freq,
                            fs=fs,
                            fault_frequency=expected_fault_frequency,
                            peak_magnitude=max_amplitude,  # 0.01,
                            decay_percentage_over_interval=0.5
                            )  # TODO: Fix peak magnitude, providing augmentation parameters?

        envelope_spectrum = ases.get_augmented_ses()

        computed = [{"envelope_spectrum": pickle.dumps({"freq": ases.frequencies,
                                                        "mag": envelope_spectrum}),
                     "augmented": True,
                     "augmentation_meta_data": {"this": "that"}}]

        new_docs = new_docs_from_computed(doc, computed)
        return new_docs

    def compute_augmentation_from_healthy_feature_doc(self, doc):
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

        # TODO: The augmentation is currently envelope spectrum specific

        if doc["severity"] != 0:
            raise ValueError("Non healthy data was used in the augmentation")

        healthy_envelope_spectrum_mag = np.array(doc["envelope_spectrum"]["mag"])
        healthy_envelope_spectrum_freq = np.array(doc["envelope_spectrum"]["freq"])

        meta_data = doc["meta_data"]
        fs = meta_data["sampling_frequency"]
        expected_fault_frequency_for_mode = meta_data["expected_fault_frequencies"]

        augmented_doc_for_mode = []
        for fault_mode in ["ball", "inner", "outer"]:
            expected_fault_frequency = expected_fault_frequency_for_mode[fault_mode]

            ases = AugmentedSES(healthy_ses=healthy_envelope_spectrum_mag,
                                healthy_ses_freq=healthy_envelope_spectrum_freq,
                                fs=fs,
                                fault_frequency=expected_fault_frequency,
                                peak_magnitude=0.01,  # max_amplitude,#
                                decay_percentage_over_interval=0.5
                                )  # TODO: Fix peak magnitude, providing augmentation parameters?

            augmented_envelope_spectrum = ases.get_augmented_ses()

            computed = {"envelope_spectrum": {"freq": list(ases.frequencies),
                                              "mag": list(augmented_envelope_spectrum)},
                        "augmented": True,
                        "augmentation_meta_data": {"this": "that"}}

            augmented_doc_for_mode.append(computed)

        new_docs = new_docs_from_computed(doc, augmented_doc_for_mode)
        return new_docs


def main(db_to_act_on):
    # db_to_act_on = "ims_test"

    db, client = make_db(db_to_act_on)
    db["augmented"].delete_many({})

    aug_obj = Augmentation(db_to_act_on)

    # Compute augmented data
    query = {"envelope_spectrum": {"$exists": True}, "severity": 0}  # TODO Severity in integers and not strings
    DerivedDoc(query, "processed", "augmented", aug_obj.compute_augmentation_from_healthy_feature_doc,
               db_to_act_on).update_database(parallel=False)

    return db["augmented"]

if __name__ == "__main__":
    r = main("phenomenological_rapid")

