from sklearn.decomposition import PCA
import numpy as np
import pickle
from dataset_management.general.update_database_with_processed import limit_frequency_components
from database_definitions import make_db


def get_trained_models(db):
    train_on_all_models = get_trained_models_train_on_all(db)

    trained_on_mode_models = get_trained_on_specific_failure_mode(db) # TODO: This is the problem case

    return train_on_all_models + trained_on_mode_models

def add_models_to_database():
    db,client = make_db()

    trained_models = get_trained_models(db)
    print(len(trained_models))
    for model in trained_models:
        doc = {"trained_object": pickle.dumps(model),
               "name": model.name,
               "implementation": "sklearn"}

        db["model"].insert_one(doc)

    return db


def get_trained_models_train_on_all(db):
    possible_severities = db["augmented"].distinct("severity")
    max_severity = possible_severities[-1]

    # Define models
    model_healthy_only = PCA(2)
    model_healthy_only.name = "healthy_only_pca"
    model_healthy_and_augmented = PCA(2)
    model_healthy_and_augmented.name = "healthy_and_augmented_pca"

    # Set up training data
    # Healthy data only
    all_healthy = [pickle.loads(doc["envelope_spectrum"])["mag"] for doc in
                   db["processed"].find({"envelope_spectrum": {"$exists": True},
                                         "augmented": False,
                                         "severity": "0"})]
    healthy_train = limit_frequency_components(np.vstack(
        all_healthy))  # Healthy data from different "modes" even though modes don't technically exist when healthy

    # # Healthy and augmented data
    all_augmented_modes = [pickle.loads(doc["envelope_spectrum"])["mag"] for doc in
                           db["augmented"].find({"envelope_spectrum": {"$exists": True},
                                                 "severity": max_severity,
                                                 "augmented": True})]
    augmented_and_healthy_train = limit_frequency_components(np.vstack(all_healthy + all_augmented_modes))

    # Train the models
    model_healthy_only.fit(healthy_train)
    model_healthy_and_augmented.fit(augmented_and_healthy_train)

    # List of trained models
    models = [model_healthy_only, model_healthy_and_augmented]
    return models


def get_trained_on_specific_failure_mode(db):
    # Define models
    failure_modes = db["augmented"].distinct("mode")

    # Create a model for each failure mode
    models = [PCA(2) for failure_mode in failure_modes]

    possible_severities = db["augmented"].distinct("severity")
    max_severity = possible_severities[-1]

    # Give each model a name
    trained_models = []
    for model, mode_name in zip(models, failure_modes):
        model.name = "PCA2_health_and_" + mode_name

        # Set up training data
        # Healthy data only
        healthy = [pickle.loads(doc["envelope_spectrum"])["mag"] for doc in
                   db["processed"].find({"envelope_spectrum": {"$exists": True},
                                         "augmented": False,
                                         "severity": "0",
                                         "mode": mode_name
                                         })]
        healthy_train = limit_frequency_components(
            np.vstack(healthy))  # Using all of the healthy data from all "modes" (even though healthy

        # Augmented data
        all_augmented_modes = [pickle.loads(doc["envelope_spectrum"])["mag"] for doc in
                               db["augmented"].find({"envelope_spectrum": {"$exists": True},
                                                     "severity": max_severity,  # Using the maximum severity only during training
                                                     "mode": mode_name,
                                                     "augmented": True})]
        # print("all augmented",len(all_augmented_modes))

        all_augmented_modes = limit_frequency_components(
            np.vstack(all_augmented_modes))  # Using all of the healthy data from all "modes" (even though healthy
        # print(all_augmented_modes[0].shape)

        augmented_and_healthy_train = np.vstack([healthy_train, all_augmented_modes])

        # # Train the models
        model.fit(augmented_and_healthy_train)
        trained_models.append(model)
    return models


def main():
    from database_definitions import model
    model.delete_many({})
    add_models_to_database()
    return model


if __name__ == "__main__":
    r = main()