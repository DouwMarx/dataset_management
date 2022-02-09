from database_definitions import raw, processed, augmented, encoding, metrics
import pickle
import numpy as np
from definitions import data_dir
from scipy.stats import ttest_ind,norm
import pickle
from sklearn.metrics import roc_curve,auc


def update_database_with_metrics():
    for doc in encoding.find({"augmented":False}): # For each measured (not augmented) encoding compute metrics

        # Get example of healthy data encoding for the same mode as the doc (Note that healthy data is not strictly associated with a given mode"
        all_healthy_encoding = [pickle.loads(enc["encoding"]) for enc in encoding.find({"model_used": doc["model_used"],
                                                                                        "severity": "0",
                                                                                        "augmented": False})]
        all_healthy_encoding = np.vstack(all_healthy_encoding)
        healthy_encoding_mean = np.mean(all_healthy_encoding, axis=0)

        # Get an example of a expected severe failure
        measured_encoding = pickle.loads(doc["encoding"])

        for expected_mode in encoding.distinct("mode"):  # For a given measurement, check its agreement with all tested failure modes

            # Get an example of what we expect severe failure would look like for this failure mode
            severe_failure_augmented_encoding = encoding.find_one({
                "augmented":True,
                "severity": "9",
                "mode": expected_mode,
                "model_used": doc["model_used"],
            })

            encoding_array = pickle.loads(severe_failure_augmented_encoding["encoding"])
            expected_damaged_mean = np.mean(encoding_array, axis=0)

            # Compute the direction between the healthy data and the expected damaged data
            direction_between_augmented_and_healthy = expected_damaged_mean-healthy_encoding_mean
            normalized_direction_between_augmented_and_healthy = direction_between_augmented_and_healthy / np.linalg.norm(direction_between_augmented_and_healthy)

            measured_projection = np.dot(measured_encoding, normalized_direction_between_augmented_and_healthy)

            healthy_projection = np.dot(all_healthy_encoding, normalized_direction_between_augmented_and_healthy)

            p,auc_score = get_metrics_for_failure_direction_projection(healthy_projection, measured_projection)

            #  Set up a dictionary of computed metrics
            metrics_dict = {"severity": doc["severity"],
                            "mode": doc["mode"],
                            "expected_mode": expected_mode,
                            "model_used": doc["model_used"],
                            "damage_direction": pickle.dumps(normalized_direction_between_augmented_and_healthy),
                            "measured_projection_in_fault_direction": {},
                            "healthy_projection_in_fault_direction": {},
                            "hypothesis_test": p,
                            "auc": auc_score
                            }

            metrics.insert_one(metrics_dict)
    return metrics

def get_metrics_for_failure_direction_projection(healthy_projection,measured_projection):
    stat, p = ttest_ind(healthy_projection, measured_projection)

    healthy_mean = np.mean(healthy_projection, axis=0)
    healthy_sd = np.std(healthy_projection, axis=0)

    sample_likelihood_measured = norm(healthy_mean, healthy_sd).pdf(measured_projection)
    sample_likelihood_healthy = norm(healthy_mean, healthy_sd).pdf(healthy_projection)

    # Compute likelihoods given the distribution
    likelihoods = np.hstack([sample_likelihood_measured, sample_likelihood_healthy])

    labels = np.hstack([np.ones(np.shape(sample_likelihood_measured)),
                        np.zeros(np.shape(sample_likelihood_healthy))])

    # Compute AUC from the RUC
    fpr, tpr, threash = roc_curve(labels, likelihoods, pos_label=0)
    auc_score = auc(fpr, tpr)

    return p,auc_score


metrics.delete_many({})
update_database_with_metrics()
