import numpy as np
from joblib import Parallel, delayed
from pymongo import MongoClient
from tqdm import tqdm

from database_definitions import make_db
from dataset_management.ultils.update_database import DerivedDoc


def process(oc,snr):
    db_og,client_og = make_db("cwr_oc" + str(oc))
    db_new,client_new = make_db("cwr_oc" + str(oc) + "_snr" + str(snr))
    db_new["raw"].drop()

    for doc in tqdm(db_og["raw"].find()):
        doc["time_series"] = list(doc["time_series"] + snr/100*np.random.normal(0,1,len(doc["time_series"])))
        db_new["raw"].insert_one(doc)

    client_og.close()
    client_new.close()

# Drop all the mongo databases that contrain "snr"
client = MongoClient()
for db in client.list_database_names():
    if "snr" in db:
        client.drop_database(db)

# Add random noise to all of the data in the dataset
for snr in [200,400,600]:
    Parallel(n_jobs=4)(delayed(process)(oc,snr) for oc in [0,1,2,3])
