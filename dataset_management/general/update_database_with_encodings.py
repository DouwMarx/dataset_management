import pickle
from database_definitions import make_db
from dataset_management.ultils.update_database import new_derived_doc, new_docs_from_computed, DerivedDoc
import torch


class Encoding():
    def __init__(self,model_query,db_to_act_on):
        self.db, self.client = make_db(db_to_act_on)
        self.model_query = model_query

        print("Computing encodings for {} models".format(self.db["model"].count_documents(model_query)))

    def compute_encoding_from_doc(self,doc):
        encodings_for_models = []
        for model in self.db["model"].find(self.model_query):
            if model["implementation"] == "torch":
                model_object = torch.load(model["path"])
            else:
                with open(model["path"], 'rb') as f:
                    model_object = pickle.load(f)

            data = torch.from_numpy(pickle.loads(doc["envelope_spectrum"])["mag"]).float()
            # limit_frequency_components(data)
            encoding = model_object.encoder(data)

            reconstruction = model_object.decoder(encoding)

            encodings_for_models.append(
                {"encoding": pickle.dumps(encoding.detach().numpy()),
                 "model_used": model["name"],
                 "model_description": model["short_description"],
                 "reconstruction": pickle.dumps(reconstruction.detach().numpy())
                 }
            )

        new_docs = new_docs_from_computed(doc, encodings_for_models)  # Add the meta-data keys
        return new_docs



def main():
    db_to_act_on = "phenomenological_rapid"
    db,client = make_db(db_to_act_on)
    db["encoding"].delete_many({})

    for augmented,source in zip([False,True],["processed","augmented"]):

        models_to_use_query = {}#{"implementation":"sklearn"} # Currently using all available models
        derived_doc_func = Encoding(models_to_use_query,db_to_act_on).compute_encoding_from_doc
        query = {"augmented": augmented, "envelope_spectrum": {"$exists": True}}
        DerivedDoc(query, source, "encoding", derived_doc_func,db_to_act_on).update_database(parallel=False)

    return db["encoding"]


if __name__ == "__main__":
    r = main()
