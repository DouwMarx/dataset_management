# This is currently done from the modelling part of the project. However, it could make sense to simply call those functions from here.
from informed_anomaly_detection.models.pca import PCAHealthyOnly, PCAHealthyAndAugmented, PCAHealthyAndMode
from database_definitions import db
from dataset_management.ultils.save_trained_models import save_trained_model

models = [PCAHealthyOnly(),PCAHealthyAndAugmented()] + [PCAHealthyAndMode(mode) for mode in ["ball","outer","inner"]]
# models = [PCAHealthyAndMode("ball")]


print(models[0].encoder)

implementation = "sklearn"
db["model"].delete_many({})
for model in models:

    insert_obj = db["model"].insert_one(
                                        {"implementation":implementation,
                                         "name":model.name,
                                         "short_description":model.short_description
                                         })

    model_id = insert_obj.inserted_id
    path = save_trained_model(model,str(model_id),model_implementation=implementation)
    db["model"].update_one({"_id": model_id}, {"$set": {"path": str(path)}})



print(db["model"].find_one())
