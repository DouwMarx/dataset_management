from database_definitions import make_db
import numpy as np
from scipy.stats import ttest_ind, norm
import pickle
from sklearn.metrics import roc_curve, auc
from dataset_management.ultils.update_database import DerivedDoc, new_docs_from_computed

def get_all_healthy_data(doc, db_to_act_on):
    db, client = make_db(db_to_act_on)
    dict_list = []

    all_healthy_ses = [pickle.loads(proc["envelope_spectrum"])["mag"] for proc in db["processed"].find(
        {
            "severity": "0",
            "envelope_spectrum": {"$exists": True},
            "augmented": False}
    )]
    all_healthy_ses = np.vstack(
        all_healthy_ses)  # TODO: This gathering of heatlhy data is probably expensive, might later be replaced with healthy test data

    # Get example of healthy data encoding for the same mode as the doc (Note that healthy data is not strictly associated with a given mode"
    all_healthy_encoding = [pickle.loads(enc["encoding"]) for enc in db["encoding"].find({
        "model_used": doc["model_used"],
        "severity": "0",
        "augmented": False})]

    all_healthy_encoding = np.vstack(all_healthy_encoding)

    healthy_encoding_mean = np.mean(all_healthy_encoding, axis=0)


# Delete me after 04022022
# def compute_reconstruction_errors(doc):
#     all_healthy_reconstruction = [pickle.loads(enc["reconstruction"]) for enc in db["encoding"].find(
#         {
#             "model_used": doc["model_used"],
#             "severity": "0",
#             "augmented": False,
#         })]
#     all_healthy_reconstruction = np.vstack(all_healthy_reconstruction)
#
#     measured_ses = db["processed"].find_one(
#         {
#             "severity": doc["severity"],
#             "mode": doc["mode"],
#             "envelope_spectrum": {"$exists": True},
#             "augmented": False
#         }
#     )
#     measured_ses = pickle.loads(measured_ses["envelope_spectrum"])["mag"]
#
#     measured_encoding = pickle.loads(doc["encoding"])
#     measured_reconstruction = pickle.loads(doc["reconstruction"])



class EncodingMovementMetrics(object):
    def __init__(self, model_used,db_to_act_on):
        self.model_used = model_used
        self.db, self.client = make_db(db_to_act_on)
        self.expected_failure_modes = self.db["encoding"].distinct("mode", {
            "augmented": True})  # What are the potential modes that have been accounted for in the augmentation.

        # Get examples of healthy and damaged data
        self.healthy_encoding = self.get_healthy_encoding_example()

        self.max_severity = self.db['augmented'].distinct("severity")[-1]

    def get_severe_augmented_encoding_example(self, expected_mode):
        # Get an example of what we expect severe failure would look like for each failure mode
        severe_augmented_encoding = self.db["encoding"].find_one(
            {
                "augmented": True,
                "severity": self.max_severity,
                "mode": expected_mode,
                "model_used": self.model_used,
            }
        )
        return pickle.loads(severe_augmented_encoding["encoding"])

    def get_healthy_encoding_example(self):
        # Get example of healthy data encoding for the same mode as the doc (Note that healthy data is not strictly associated with a given mode"
        all_healthy_encoding = [pickle.loads(enc["encoding"]) for enc in self.db["encoding"].find({
            "model_used": self.model_used,
            "severity": "0",
            "augmented": False})]

        if len(all_healthy_encoding) == 0:
            print("No documents found")


        return np.vstack(all_healthy_encoding)

    def get_expected_failure_direction(self, expected_mode):
        expected_mode_encoding = self.get_severe_augmented_encoding_example(expected_mode)
        expected_mode_encoding_mean = np.mean(expected_mode_encoding, axis=0)

        healthy_encoding_mean = np.mean(self.healthy_encoding, axis=0)

        # Compute the direction between the healthy data and the expected damaged data
        direction_healthy_to_augmented = expected_mode_encoding_mean - healthy_encoding_mean
        normalized_direction_healthy_to_augmented = direction_healthy_to_augmented / np.linalg.norm(
            direction_healthy_to_augmented)
        return normalized_direction_healthy_to_augmented

    def get_projection_onto_failure_direction(self, measured_encoding, normalized_direction_healthy_to_augmented):
        measured_projection = np.dot(measured_encoding, normalized_direction_healthy_to_augmented)
        # healthy_projection = np.dot(self.healthy_encoding, normalized_direction_healthy_to_augmented)
        return measured_projection

    def get_metrics_for_failure_direction_projection(self, measured_encoding, expected_mode):
        normalized_direction_healthy_to_augmented = self.get_expected_failure_direction(expected_mode)

        measured_projection = self.get_projection_onto_failure_direction(measured_encoding,
                                                                         normalized_direction_healthy_to_augmented)
        healthy_projection = self.get_projection_onto_failure_direction(self.healthy_encoding,
                                                                        normalized_direction_healthy_to_augmented)

        stat, p = ttest_ind(healthy_projection, measured_projection)

        healthy_mean = np.mean(healthy_projection, axis=0)
        healthy_sd = np.std(healthy_projection, axis=0)

        sample_likelihood_measured = norm(healthy_mean, healthy_sd).pdf(measured_projection)
        sample_likelihood_healthy = norm(healthy_mean, healthy_sd).pdf(healthy_projection)

        # Compute likelihoods given the distribution
        likelihoods = np.hstack([sample_likelihood_measured, sample_likelihood_healthy])

        labels = np.hstack([np.ones(np.shape(sample_likelihood_measured)),
                            np.zeros(np.shape(sample_likelihood_healthy))])

        # Compute AUC from the ROC
        fpr, tpr, threash = roc_curve(labels, likelihoods, pos_label=0)
        auc_score = auc(fpr, tpr)

        return p, auc_score, normalized_direction_healthy_to_augmented, measured_projection, healthy_projection

    def compute_metrics_from_doc(self, doc):
        measured_encoding = pickle.loads(doc["encoding"])
        metric_dicts = []
        for expected_mode in self.expected_failure_modes:
            (p,
             auc_score,
             normalized_direction_healthy_to_augmented,
             measured_projection,
             healthy_projection
             ) = self.get_metrics_for_failure_direction_projection(measured_encoding, expected_mode)

            #  Set up a dictionary of computed metrics
            metrics_dict = {
                "expected_mode": expected_mode,
                "model_used": doc["model_used"],
                "damage_direction": pickle.dumps(normalized_direction_healthy_to_augmented),
                "measured_projection_in_fault_direction": pickle.dumps(measured_projection),
                "healthy_projection_in_fault_direction": pickle.dumps(healthy_projection),
                "hypothesis_test": p,
                "auc": auc_score
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

def main():
    db_to_act_on = "phenomenological_rapid"
    db,client = make_db(db_to_act_on)
    db["metrics"].delete_many({})

    # Compute the metrics
    # model_used = "healthy_only_pca"
    for model_used in db["encoding"].distinct("model_used"):
        print(model_used)
        query = {"augmented": False, "model_used": model_used}
        em_obj = EncodingMovementMetrics(model_used,db_to_act_on)
        # models: encoding.distinct("model_used")
        DerivedDoc(query, "encoding", "metrics", em_obj.compute_metrics_from_doc,db_to_act_on).update_database(parallel=False)
    return db["metrics"]

if __name__ == "__main__":
    r = main()
