# This is currently done from the modelling part of the project. However, it could make sense to simply call those functions from here.
from informed_anomaly_detection.models.pca import PCAHealthyOnly, PCAHealthyAndAugmented, PCAHealthyAndMode
from informed_anomaly_detection.models.prescribe_latent_movement_direction import get_prescribe_latent_movement_direction_model
from informed_anomaly_detection.models.maximize_distance_between_latent_directions import get_maximize_distance_between_latent_directions_model
from informed_anomaly_detection.models.maximize_distance_between_latent_directions_with_magnitude import get_maximize_distance_between_latent_directions_with_magnitude
from database_definitions import db
from dataset_management.ultils.save_trained_models import save_trained_model



def main():
    db["model"].delete_many({})

    # sklearn_models = [PCAHealthyOnly(), PCAHealthyAndAugmented()] + [PCAHealthyAndMode(mode) for mode in
    #                                                                  ["ball", "outer", "inner"]]

    # implementation = "sklearn"
    # for model in sklearn_models:
    #     insert_obj = db["model"].insert_one(
    #         {"implementation": implementation,
    #          "name": model.name,
    #          "short_description": model.short_description
    #          })
    #
    #     model_id = insert_obj.inserted_id
    #     path = save_trained_model(model, str(model_id), model_implementation=implementation)
    #     db["model"].update_one({"_id": model_id}, {"$set": {"path": str(path)}})


    torch_models = [get_prescribe_latent_movement_direction_model(),
                    get_maximize_distance_between_latent_directions_model(),
                    get_maximize_distance_between_latent_directions_with_magnitude(),
                    ]

    implementation = "torch"
    for model in torch_models:
        insert_obj = db["model"].insert_one(
            {"implementation": implementation,
             "name": model.name,
             "short_description": model.short_description
             })

        model_id = insert_obj.inserted_id
        path = save_trained_model(model, str(model_id), model_implementation=implementation)
        db["model"].update_one({"_id": model_id}, {"$set": {"path": str(path)}})
    return db["model"]

if __name__ == "__main__":
    main()