from database_definitions import make_db
from dataset_management.ultils.save_trained_models import save_trained_model
from informed_anomaly_detection.models.env_spec_auto_encoder_latent import get_trained_latent_separation_model, \
    get_trained_latent_separation_for_intermediate_states_during_training


def train_model(db_to_act_on,model_training_parameters):
    db, client = make_db(db_to_act_on)
    db["model"].delete_many({})

    model = get_trained_latent_separation_model(db_to_act_on,**model_training_parameters)

    implementation = "torch"
    insert_obj = db["model"].insert_one(
        {"implementation": implementation,
         "name": model.name,
         "short_description": model.short_description
         })
    model_id = insert_obj.inserted_id
    path = save_trained_model(model, str(model_id), model_implementation=implementation)
    db["model"].update_one({"_id": model_id}, {"$set": {"path": str(path)}})
    return db["model"]

def train_model_and_keep_the_intermediate_states_of_training(db_to_act_on,model_training_parameters):
    db, client = make_db(db_to_act_on)
    db["model"].delete_many({})

    models = get_trained_latent_separation_for_intermediate_states_during_training(db_to_act_on,**model_training_parameters)

    for model in models:
        implementation = "torch"
        insert_obj = db["model"].insert_one(
            {"implementation": implementation,
             "name": model.name,
             "short_description": model.short_description
             })
        model_id = insert_obj.inserted_id
        path = save_trained_model(model, str(model_id), model_implementation=implementation)
        db["model"].update_one({"_id": model_id}, {"$set": {"path": str(path)}})
    return db["model"]
# TODO package the part that is used to insert the entry in the database.

def main(db_to_act_on):
    train_model(db_to_act_on)

if __name__ == "__main__":
    main("phenomenological")