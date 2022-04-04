from augment_data.envelope_with_traingular_spikes.make_envelope_spectrum_triangular import \
    AugmentedEnvelopeFromFaultFrequencies,AugmentedEnvelopeImprovedTriangular
from database_definitions import make_db
from dataset_management.ultils.update_database import DerivedDoc, new_docs_from_computed
import numpy as np


class Augmentation():
    def __init__(self, db_to_act_on):
        self.db_to_act_on = db_to_act_on
        self.db, self.client = make_db(
            db_to_act_on)
        self.max_severity = self.db["processed"].distinct("severity")[-1]

        self.failure_modes = ["ball", "inner", "outer"]

        # TODO: Very important, Notice that setting the augmented amplitude using failure data is cheating: This is however now done for rapid iteration.
        # However, there will always be a discrepancy in practice.

        self.env_spec_max_for_mode = {}
        for mode in self.db["processed"].distinct("mode"):
            docs = self.db["processed"].find(
                {"envelope_spectrum": {"$exists": True},
                 "mode": mode
                 }).sort("severity", direction=-1).limit(
                20)  # Use the final 20 documents to estimate the maximum severity
            env_spec_max = [np.max(doc["envelope_spectrum"]["mag"]) for doc in docs]
            self.env_spec_max_for_mode.update({mode: np.median(env_spec_max)})

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

        # Instantiate the triagle at frequency augmentation

        augmented_doc_for_mode = []
        # Loop through the different failure modes that could appear
        ases = AugmentedEnvelopeImprovedTriangular(healthy_envelope_spectrum_freq,)
        for fault_mode in ["ball", "inner", "outer"]:
            expected_fault_frequency = expected_fault_frequency_for_mode[fault_mode]

            if fault_mode in self.env_spec_max_for_mode:
                peak_mag = self.env_spec_max_for_mode[
                    fault_mode]  # TODO: notice that knowing the expected magnitude is cheating.
            else:
                peak_mag = self.env_spec_max_for_mode[max(self.env_spec_max_for_mode)]

            # ases = AugmentedEnvelopeFromFaultFrequencies(healthy_ses=healthy_envelope_spectrum_mag,
            #                                              healthy_ses_freq=healthy_envelope_spectrum_freq,
            #                                              fs=fs,
            #                                              fault_frequency=expected_fault_frequency,
            #                                              # peak_magnitude=0.04,  # max_amplitude,#
            #                                              peak_magnitude=peak_mag,  # max_amplitude,#
            #                                              decay_percentage_over_interval=0.999
            #                                              )  # TODO: Fix peak magnitude, providing augmentation parameters?

            augmented_envelope_spectrum = ases.get_augmented_ses(healthy_envelope_spectrum_mag,peak_mag,expected_fault_frequency)

            computed = {"envelope_spectrum": {"freq": list(ases.frequencies),
                                              "mag": list(augmented_envelope_spectrum)},
                        "augmented": True,
                        "meta_data": ases.augmentation_meta_data,
                        "mode": fault_mode,
                        "severity": 999}  # Use 999 severity for now

            augmented_doc_for_mode.append(computed)

        new_docs = new_docs_from_computed(doc, augmented_doc_for_mode)  # Add the meta-data keys
        return new_docs


def ims_outer_t2_c1_aug(db_to_act_on):
    db, client = make_db(db_to_act_on)
    db["augmented"].delete_many({})

    aug_obj = Augmentation(db_to_act_on)

    # Compute augmented data
    query = {"envelope_spectrum": {"$exists": True},
             "severity": 0,
             'ims_test_number': "2",
             'ims_channel_number': "1",
             }
    DerivedDoc(query, "processed", "augmented", aug_obj.compute_augmentation_from_healthy_feature_doc,
               db_to_act_on).update_database(parallel=False)

    return db["augmented"]


def main(db_to_act_on):
    db, client = make_db(db_to_act_on)
    db["augmented"].delete_many({})

    aug_obj = Augmentation(db_to_act_on)

    # Compute augmented data
    query = {"envelope_spectrum": {"$exists": True}, "severity": 0}
    DerivedDoc(query, "processed", "augmented", aug_obj.compute_augmentation_from_healthy_feature_doc,
               db_to_act_on).update_database(parallel=False)

    return db["augmented"]


if __name__ == "__main__":
    r = main("phenomenological")
