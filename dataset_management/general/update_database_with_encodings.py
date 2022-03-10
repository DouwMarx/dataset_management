import pickle
from database_definitions import make_db
from dataset_management.ultils.update_database import new_derived_doc, new_docs_from_computed, DerivedDoc
import torch


class Encoding():
    def __init__(self,model_query,db_to_act_on):
        self.db, self.client = make_db(db_to_act_on)
        self.model_query = model_query

        print("Computing encodings for {} models".format(self.db["model"].count_documents(model_query)))

        torch_query = self.model_query.copy().update({"implementation":"torch"})
        self.torch_docs = list(self.db["model"].find(torch_query))
        self.torch_models = [torch.load(model["path"]).to("cpu") for model in self.torch_docs]

        # sklearn_query = self.model_query.copy().update({"implementation":"sklearn"})
        # self.torch_models = [torch.load(model["path"]).to("cpu") for model in self.db["model"].find(sklearn_query)]
    #     self.sklearn_models = [torch.load(model["path"]).to("cpu") for model in self.db["model"].find(self.model_query)]
    # else:
    # with open(model["path"], 'rb') as f:
    #     model_object = pickle.load(f)

    def compute_encoding_from_doc(self,doc):
        # TODO: This should be in the init otherwise model is loaded the whole time
        encodings_for_models = []
        # if doc["implementation"] == "torch":
        data = torch.tensor(doc["envelope_spectrum"]["mag"])
        for model_object,model_doc in zip(self.torch_models,self.torch_docs):

            encoding = model_object.encoder(data)
            reconstruction = model_object.decoder(encoding)

            encodings_for_models.append(
                {"encoding": encoding.tolist(),
                 "model_used": model_doc["name"],
                 "model_description": model_doc["short_description"],
                 "reconstruction": reconstruction.tolist()
                 }
            )

        new_docs = new_docs_from_computed(doc, encodings_for_models)  # Add the meta-data keys
        return new_docs



def main(db_to_act_on):
    db,client = make_db(db_to_act_on)
    db["encoding"].delete_many({})

    for source in ["processed","augmented"]:

        models_to_use_query = {}#{"implementation":"sklearn"} # Currently using all available models
        derived_doc_func = Encoding(models_to_use_query,db_to_act_on).compute_encoding_from_doc
        # TODO: There should be no augmented data in the processed db?
        query = {"envelope_spectrum": {"$exists": True}}
        DerivedDoc(query, source, "encoding", derived_doc_func,db_to_act_on).update_database(parallel=False)

    return db["encoding"]


if __name__ == "__main__":
    r = main("phenomenological_rapid")
