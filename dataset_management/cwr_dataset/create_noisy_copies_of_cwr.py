import numpy as np
from joblib import Parallel, delayed
from pymongo import MongoClient
from tqdm import tqdm

from database_definitions import make_db
from dataset_management.ultils.update_database import DerivedDoc

# See below for snr definition
# https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python


def process(doc, snr_levels=None):
    if snr_levels is None:
        # snr_levels = [3, 4, 5, 6]
        snr_levels = [4, 8, 12, 16,20]
    db,client = make_db("cwr")
    docs = []
    for snr in snr_levels:
        # Create a copy of the document
        new_doc = doc.copy()
        new_doc.pop("_id")
        # Remove the id field
        new_doc.update({"snr": snr})
        doc["time_series"] = list(doc["time_series"] + np.random.normal(0, snr**2, len(doc["time_series"])))
        docs.append(new_doc)

    db["raw"].insert_many(docs)
    client.close()

db,client = make_db("cwr")

# First remove all entries in the database that has snr > 0
db["raw"].delete_many({"snr": {"$gt": 0}})


Parallel(n_jobs=6)(delayed(process)(doc) for doc in tqdm(db["raw"].find({"snr":0})))

# for doc in tqdm(db["raw"].find({"snr":0})):
#     process(doc)

    # # Use only the data with snr level 0 as the source
    # for doc in tqdm(db_og["raw"].find({"snr":0})):
