
from database_definitions import make_db
import random
from tqdm import tqdm

def test_train_split(db_to_act_on, test_fraction=0.1):
    db, client = make_db(db_to_act_on)
    healthy_ids = list(db["raw"].find({"severity": 0}, projection="_id"))
    if len(healthy_ids)==0:
        raise ValueError("No healthy data that can be labeled as test or train")
    else:
        print("Updating the train vs test labels for {} samples".format(len(healthy_ids)))


    n_healthy = len(healthy_ids)

    test_set_size = int(test_fraction * n_healthy)
    train_set_size = n_healthy - test_set_size

    labels = ["test"] * test_set_size + ["train"] * train_set_size
    random.shuffle(labels)  # Shuffle the labels for train or test

    for id, label in tqdm(zip(healthy_ids, labels)):
        db["raw"].update_one({'_id': id["_id"]}, {"$set": {"set": label}})

    return db
