import numpy as np
from joblib import Parallel, delayed
from pymongo import MongoClient
from tqdm import tqdm

from database_definitions import make_db
from dataset_management.ultils.update_database import DerivedDoc


"""
Some fault seeded experiments have very clear fault signatures for certain tests. This script will create copies of the CWR dataset with different levels of noise to make the fault detection more difficult and ensure that it is easier to differentiate between the performance of different algorithms.
"""

# See below for snr definition
# https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
def get_median_variance(dataset_name):
    # Compute the average variance of all the time series in the database
    db,client = make_db(dataset_name)
    variances = []
    for doc in tqdm(db["raw"].find({"snr":0})):
        centered = doc["time_series"] - np.mean(doc["time_series"])
        variances.append(np.var(centered))
    client.close()
    median_variance = np.median(variances)
    print("Median variance: {}".format(median_variance))
    return median_variance

def process(db_name,doc, snr_levels=None,median_variance=0.08582927368282073): # TODO: Automate median variance computation
    if snr_levels is None:
        snr_levels = np.logspace(-2, 0, 3)

    noise_variance = median_variance / snr_levels
    noise_sd = np.sqrt(noise_variance)

    db,client = make_db(db_name)
    docs = []
    for sd,snr in zip(noise_sd,snr_levels):
        # Create a copy of the document
        new_doc = doc.copy()
        new_doc.pop("_id") # Remove the id field so that mongo will create a new one
        new_doc.update({"snr": float(snr)}) # Update the snr field with the new value

        time_series = np.array(doc["time_series"])
        new_doc["time_series"] = list(time_series + np.random.normal(0, sd, size=time_series.size))# Notice numpy takes the standard deviation as argument
        docs.append(new_doc)

    db["raw"].insert_many(docs)
    client.close()

def add_noisy_copies_of_dataset(dataset_name,snr_levels=None,parallel=True):
    db,client = make_db("cwr")

    # First remove all entries in the database that has snr > 0 (Keep only the raw clean,unmodified data)
    db["raw"].delete_many({"snr": {"$ne": 0}})

    # We compute the median variance of all the time series in the database
    median_variance = get_median_variance(dataset_name)  # about 0.085, hard coded into process function keyword argument
    print("Median variance for {} is : {}".format(dataset_name,median_variance))

    # Run the process function on all the documents in the database


    if parallel:
        Parallel(n_jobs=10)(delayed(process)( dataset_name,doc,snr_levels,median_variance) for doc in tqdm(list(db["raw"].find({"snr":0}))))
    else:
        for doc in list(db["raw"].find({"snr":0})):
            process(dataset_name, doc = doc,snr_levels=snr_levels,median_variance=median_variance)


db_name = "lms"
add_noisy_copies_of_dataset(db_name,snr_levels=np.logspace(-2, 0, 3),parallel=True)