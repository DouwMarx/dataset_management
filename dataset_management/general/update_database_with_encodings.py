import pickle
from database_definitions import make_db
from dataset_management.general.update_database_with_processed import limit_frequency_components
from dataset_management.ultils.update_database import new_derived_doc, new_docs_from_computed, DerivedDoc


def compute_encoding_from_doc(doc):
    # TODO: make sure encodings are computed for both real and augmented data
    db,client = make_db()
    models = [pickle.loads(doc["model_object"]) for doc in db["model"].find()]
    # print("models",models[0],len(models))
    encodings_for_models = []
    for model in models:
        encoding = model.transform(limit_frequency_components(pickle.loads(doc["envelope_spectrum"])["mag"]))

        reconstruction = model.inverse_transform(encoding)

        encodings_for_models.append(
            {"encoding": pickle.dumps(encoding),
             "model_used": model.name,
             "reconstruction": pickle.dumps(reconstruction)
             }
        )

    new_docs = new_docs_from_computed(doc,encodings_for_models)  # Add the meta-data keys
    return new_docs


def main():
    from database_definitions import encoding

    encoding.delete_many({})

    # # Apply encoding for both augmented and not augmented data
    query = {"augmented": True}
    DerivedDoc(query, "augmented", "encoding", compute_encoding_from_doc).update_database(parallel=True)

    query = {"augmented": False, "envelope_spectrum": {"$exists": True}}
    DerivedDoc(query, "processed", "encoding", compute_encoding_from_doc).update_database(parallel=True)

    return encoding


if __name__ == "__main__":
    r = main()
