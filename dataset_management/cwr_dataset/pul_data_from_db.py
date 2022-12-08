import pathlib
from itertools import product

import numpy as np
import pandas as pd

from dataset_management.ultils.write_data_in_standard_format import export_data_to_file_structure


def get_engineering_feature_data_from_db(oc, sev, snr):
    # Note that features to use is not really required after exporting the data from the database and just loading it from the pickle file
    features_to_use = [
        'rms',
        'sra',
        'kurtosis',
        'crest_factor',
        # 'skewness',
        'ball_h2',
        # 'ball_h3',
        'ball_h4',
        # 'ball_h5',
        'outer_h2',
        # 'outer_h3',
        'outer_h4',
        # 'outer_h5',
        'inner_h2',
        # 'inner_h3',
        'inner_h4',
        # 'inner_h5',
        'entropy',
        'spectral_entropy'
    ]
    # Get the data with specific fields from the database
    filter = {key: 1 for key in
              features_to_use}  # The filter that will extract only the relevant fields from the database
    filter.update({"_id": 0})  # Don't extract the id field

    # Pull the data from the database
    from pymongo import MongoClient
    client = MongoClient()
    db = client["cwr"]
    collection = db["raw"]

    # Extract the healthy data
    healthy_data = pd.DataFrame(
        list(collection.find(
            {"oc": oc,
             "severity": 0.0,
             "snr": snr,
             "mode": "healthy",
             },
            filter)
        )
    )
    if len(healthy_data) == 0:  # Skip if there is no data
        raise ValueError("No healthy data found in the database for operating condition " + str(oc))
    # else:
        # print("Found {} healthy data points".format(len(healthy_data)))

    # Find the faulty data for each fault mode

    faulty_test_datasets = {}
    for fault_mode in "ball", "outer centre", "inner":
        faulty_test_data = pd.DataFrame(
            list(collection.find(
                {"oc": oc,
                 "severity": sev,
                 "snr": snr,
                 "mode": fault_mode,
                 }
                , filter)
            ))
        faulty_test_datasets[fault_mode] = faulty_test_data

        print_string = "OC: " + str(oc) + " SNR: " + str(snr) + " SEV: " + str(sev)
        if len(faulty_test_data) == 0:  # Skip if there is no data
            print("No faulty data for " + print_string)
            return None, None  # Stop the analysis if there is no faulty data
    return healthy_data, faulty_test_datasets

def get_frequency_feature_data_from_db(oc, sev, snr):

    # Pull the data from the database
    from pymongo import MongoClient
    client = MongoClient()
    db = client["cwr"]
    collection = db["raw"]

    # Extract the healthy data
    healthy_data = list(collection.find(
        {"oc": oc,
         "severity": 0.0,
         "snr": snr,
         "mode": "healthy",
         },
        {"fft": 1, "_id": 0}  # Only extract the fft field
    )
    )

    if len(healthy_data) == 0:  # Skip if there is no data
        raise ValueError("No healthy frequency data found in the database for operating condition " + str(oc))
    else:
        print("Found {} healthy data points".format(len(healthy_data)))

    healthy_data = pd.DataFrame([d["fft"] for d in healthy_data])


    faulty_test_datasets = {}
    for fault_mode in "ball", "outer centre", "inner":
        faulty_test_data = list(collection.find(
            {"oc": oc,
             "severity": sev,
             "snr": snr,
             "mode": fault_mode,
             }
            ,
            {"fft": 1, "_id": 0}  # Only extract the fft field
        )
        )

        faulty_test_data = pd.DataFrame([d["fft"] for d in faulty_test_data])

        faulty_test_datasets[fault_mode] = faulty_test_data

        print_string = "OC: " + str(oc) + " SNR: " + str(snr) + " SEV: " + str(sev)
        if len(faulty_test_data) == 0:  # Skip if there is no data
            print("No frequency faulty data for " + print_string)
            return None, None  # Stop the analysis if there is no faulty data
    return healthy_data, faulty_test_datasets



from pymongo import MongoClient
client = MongoClient()
db = client["cwr"]
collection = db["raw"]

for oc, sev, snr in product(collection.distinct("oc"), collection.distinct("severity"), collection.distinct("snr")):
    # Get the healthy data
    for extract_function, data_type in zip([get_engineering_feature_data_from_db, get_frequency_feature_data_from_db], ["engineering", "frequency"]):
        healthy, faulty_data_dict= extract_function(oc, sev, snr)
        if healthy is not None:
            print("for OC: {} SNR: {} SEV: {}, lengths are: healthy {}, faulty {}".format(oc, snr, sev, len(healthy), [len(faulty) for faulty in faulty_data_dict.values()]))

            ground_truth_fault_dir = pd.concat(list(faulty_data_dict.values()), ignore_index=True).mean()-healthy.mean()
            ground_truth_fault_dir = ground_truth_fault_dir/np.linalg.norm(ground_truth_fault_dir)
            print("Ground truth fault direction for {}: {}".format(data_type, ground_truth_fault_dir.transpose()))
            print("")

            name = 'cwr_{}_oc{}_snr{}_sev{}'.format(data_type,oc, snr, sev)
            export_data_to_file_structure(dataset_name=name,
                                          healthy_data=healthy,
                                          faulty_data_dict=faulty_data_dict,
                                          export_path=pathlib.Path(
                                              "/home/douwm/projects/PhD/code/biased_anomaly_detection/data"),
                                          metadata={'ground_truth_fault_direction': list(ground_truth_fault_dir),
                                                    'dataset_name': name}
                                          )


