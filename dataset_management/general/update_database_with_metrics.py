from itertools import combinations

from database_definitions import make_db
import numpy as np
from scipy.stats import ttest_ind, norm
from sklearn.metrics import roc_curve, auc
from dataset_management.ultils.update_database import DerivedDoc, new_docs_from_computed


class EncodingMovementMetrics(object):
    def __init__(self, model_used, db_to_act_on):
        self.model_used = model_used
        self.db, self.client = make_db(db_to_act_on)
        self.expected_failure_modes = self.db["encoding"].distinct("mode", {
            "augmented": True})  # What are the potential modes that have been accounted for in the augmentation.
        self.max_severity = self.db['augmented'].distinct("severity")[-1]

        # Get examples of healthy and damaged data
        self.healthy_encoding = self.get_healthy_encoding_examples()
        self.severe_augmented_data_examples_per_mode = {mode: self.get_severe_augmented_data_example(mode) for mode in
                                                        self.expected_failure_modes}

        # Get the medians of the healthy and damaged distributions so that the failure directions can be computed
        self.healthy_encoding_median = np.median(self.healthy_encoding, axis=0)
        self.severe_augmented_data_median_per_mode = {key: np.median(val, axis=0) for key, val in
                                                      self.severe_augmented_data_examples_per_mode.items()}

        # Get the expected failure directions
        self.expected_failure_directions_per_mode = {mode: self.get_expected_failure_direction(mode) for mode in
                                                     self.expected_failure_modes}

        # Get the healthy projections onto the failure directions
        self.healthy_projection_per_mode = {
            mode:self.get_projection_onto_failure_direction(self.healthy_encoding,
                                                            self.expected_failure_directions_per_mode[mode])
            for mode in self.expected_failure_modes}


        self.fitted_healthy_projection_per_mode = {}
        for mode in self.expected_failure_modes:
            mu,std = norm.fit(self.healthy_projection_per_mode[mode])
            self.fitted_healthy_projection_per_mode.update({mode:{"mu":mu,
                                                                 "std":std}})


    def check_latent_constraints(self):
        """
        Checks if the constraints imposed during training is was succesfull/ sufficiently applied
        """
        print(" ")
        print("Checks on latent movement")
        print("Latent movement directions between cluster medians:")
        combinations_of_modes = list(combinations(["inner", "outer", "ball"], 2))
        combinations_of_modes = [list(comb) for comb in combinations_of_modes]

        for combination in combinations_of_modes:
            print("dot prod", combination, np.dot(self.expected_failure_directions_per_mode[combination[0]],
                                                  self.expected_failure_directions_per_mode[combination[1]]))
        for combination in combinations_of_modes:
            print("magnitude", combination, np.linalg.norm(
                self.severe_augmented_data_median_per_mode[combination[0]] - self.severe_augmented_data_median_per_mode[
                    combination[1]]))

    def get_severe_augmented_data_example(self, expected_mode):
        # Get an example of what we expect severe failure would look like for each failure mode
        severe_augmented_encodings = [enc["encoding"] for enc in self.db["encoding"].find(
            {
                "augmented": True,
                "severity": 999,  # self.max_severity,
                "mode": expected_mode,
                "model_used": self.model_used,
            }
        )]

        if len(severe_augmented_encodings) == 0:
            print("No augmented encoding documents found")
        else:
            print(
                "Making use of {} augmented samples for mode {}".format(len(severe_augmented_encodings), expected_mode))

        return severe_augmented_encodings

    def get_healthy_encoding_examples(self):
        # Get example of healthy data encoding for the same mode as the doc (Note that healthy data is not strictly associated with a given mode"
        all_healthy_encoding = [enc["encoding"] for enc in self.db["encoding"].find({
            "model_used": self.model_used,
            "severity": 0,
            "augmented": False})]

        if len(all_healthy_encoding) == 0:
            print("No healthy encoding documents found")
        else:
            print(
                "Making use of {} healthy samples defining the healthy distribution".format(len(all_healthy_encoding)))

        return np.vstack(all_healthy_encoding)

    def get_expected_failure_direction(self, expected_mode):
        # Compute the direction between the healthy data and the expected damaged data
        direction_healthy_to_augmented = self.severe_augmented_data_median_per_mode[
                                             expected_mode] - self.healthy_encoding_median
        normalized_direction_healthy_to_augmented = direction_healthy_to_augmented / np.linalg.norm(
            direction_healthy_to_augmented)
        return normalized_direction_healthy_to_augmented

    def get_projection_onto_failure_direction(self, measured_encoding, normalized_direction_healthy_to_augmented):
        measured_projection = np.dot(measured_encoding, normalized_direction_healthy_to_augmented)
        # healthy_projection = np.dot(self.healthy_encoding, normalized_direction_healthy_to_augmented)
        return measured_projection

    def get_metrics_for_failure_direction_projection(self, measured_encoding, expected_mode):

        measured_projection = self.get_projection_onto_failure_direction(measured_encoding,self.expected_failure_directions_per_mode[expected_mode])
        mu = self.fitted_healthy_projection_per_mode[expected_mode]["mu"]
        std = self.fitted_healthy_projection_per_mode[expected_mode]["std"]
        p = norm.pdf(measured_projection, mu, std)


        # Computation of the AUC cannot be done on a per saple basis?
        # sample_likelihood_measured = norm(healthy_mean, healthy_sd).pdf(measured_projection)
        # sample_likelihood_healthy = norm(healthy_mean, healthy_sd).pdf(healthy_projection)
        #
        # # Compute likelihoods given the distribution
        # likelihoods = np.hstack([sample_likelihood_measured, sample_likelihood_healthy])
        #
        # labels = np.hstack([np.ones(np.shape(sample_likelihood_measured)),
        #                     np.zeros(np.shape(sample_likelihood_healthy))])
        #
        # # Compute AUC from the ROC
        # fpr, tpr, threash = roc_curve(labels, likelihoods, pos_label=0)
        # auc_score = auc(fpr, tpr)

        return measured_projection, p

    def compute_metrics_from_doc(self, doc):
        measured_encoding = doc["encoding"]
        metric_dicts = []
        for expected_mode in self.expected_failure_modes:
            (measured_projection,
             p,
             ) = self.get_metrics_for_failure_direction_projection(measured_encoding, expected_mode)

            # print("mp :",measured_projection)
            # print("hp:",healthy_projection)

            #  Set up a dictionary of computed metrics
            metrics_dict = {
                "expected_mode": expected_mode,
                # "damage_direction": list(normalized_direction_healthy_to_augmented),
                "measured_projection_in_fault_direction": measured_projection,
                # "healthy_projection_in_fault_direction": list(healthy_projection),
                "p": p,
                "neg_log_p": -np.log(p),
                # "auc": auc_score
            }

            metric_dicts.append(metrics_dict)

        new_docs = new_docs_from_computed(doc, metric_dicts)  # Add the meta-data keys

        return new_docs


# a = EncodingMovementMetrics('healthy_only_pca')
# print(a.get_severe_augmented_encoding_example("ball"))
# print(a.get_expected_failure_direction("ball"))
#
# db,client = make_db()
#
# encoding_example = db["encoding"].find_one()
# encoding_example = pickle.loads(encoding_example["encoding"])
#
# tup = a.get_metrics_for_failure_direction_projection(encoding_example,"ball")

def main(db_to_act_on):
    db, client = make_db(db_to_act_on)
    db["metrics"].delete_many({})
    # Compute the metrics
    for model_used in db["encoding"].distinct("model_used"):
        print("Computing metrics for model: ", model_used)
        query = {"augmented": False, "model_used": model_used}  # Applying metrics only to non-augmented data
        em_obj = EncodingMovementMetrics(model_used, db_to_act_on)
        DerivedDoc(query, "encoding", "metrics", em_obj.compute_metrics_from_doc,db_to_act_on).update_database(parallel=False)
    return db


if __name__ == "__main__":
    db_to_act_on = "phenomenological"
    r = main(db_to_act_on)
