import numpy as np
from joblib import Parallel, delayed
from pymongo import MongoClient
from tqdm import tqdm

from database_definitions import make_db
from dataset_management.ultils.update_database import DerivedDoc


"""
The CWR dataset has very clear fault signatures for certain tests. This script will create copies of the CWR dataset with different levels of noise to make the fault detection more difficult and ensure that it is easier to differentiate between the performance of different algorithms.
"""

# See below for snr definition
# https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
def get_median_variance():
    # Compute the average variance of all the time series in the database
    db,client = make_db("cwr")
    variances = []
    for doc in tqdm(db["raw"].find({"snr":0})):
        centered = doc["time_series"] - np.mean(doc["time_series"])
        variances.append(np.var(centered))
    client.close()
    median_variance = np.median(variances)
    print("Median variance: {}".format(median_variance))
    return median_variance

def process(doc, snr_levels=None,median_variance=19.145314017798253): # TODO: Automate median variance computation
    if snr_levels is None:
        # snr_levels = np.array([0.1])
        snr_levels = np.logspace(-2, 0, 3)
    # SNR = P_signal / P_noise
    # P_noise = P_signal / SNR
    noise_variance = median_variance / snr_levels
    noise_sd = np.sqrt(noise_variance)

    # print("SNR levels: {}".format(snr_levels))
    # print("Noise SD: {}".format(noise_sd))

    db,client = make_db("cwr")
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

db,client = make_db("cwr")

# First remove all entries in the database that has snr > 0 (Keep only the raw clean,unmodified data)
# db["raw"].delete_many({"snr": {"$gt": 0}})
db["raw"].delete_many({"snr": {"$ne": 0}})

# We compute the median variance of all the time series in the database
median_variance = get_median_variance()  # about 0.085, hard coded into process function keyword argument
print("Median variance: {}".format(median_variance))

# Run the process function on all the documents in the database

parallel =True
if parallel:
    Parallel(n_jobs=10)(delayed(process)(doc) for doc in tqdm(db["raw"].find({"snr":0})))
else:
    for doc in list(db["raw"].find({"snr":0})):
        process(doc)


