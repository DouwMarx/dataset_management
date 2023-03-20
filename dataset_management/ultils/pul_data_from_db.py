import pathlib
from itertools import product
import pandas as pd
from pymongo import MongoClient

from dataset_management.ultils.write_data_in_standard_format import export_data_to_file_structure


# Script pulls data from the database and exports it to the standard file structure on local pc
def get_data_from_db(oc, sev, snr, db_name, filter_to_use):
    # Pull the data from the database
    from pymongo import MongoClient
    client = MongoClient()
    db = client[db_name]
    collection = db["raw"]

    # Extract the healthy data
    healthy_data = list(collection.find(
            {"oc": oc,
             "severity": 0,
             "snr": snr,
             "mode": "healthy",
             },
            filter_to_use)
        )
    if len(healthy_data) == 0:  # Skip if there is no data
        print("No healthy engineering feature data found in the database for operating condition oc: " + str(oc) + " snr: " + str(snr) + " sev: " + str(sev))
        return None, None
    else:
        print("Found {} healthy data points".format(len(healthy_data)))

    # Find the corresponding faulty data for each fault mode
    faulty_test_datasets = {}
    for fault_mode in "ball", "outer", "inner":
        faulty_test_data = list(collection.find(
                {"oc": oc,
                 "severity": sev,
                 "snr": snr,
                 "mode": fault_mode,
                 }
                , filter_to_use)
            )
        faulty_test_datasets[fault_mode] = faulty_test_data

        if len(faulty_test_data) == 0:  # Skip if there is no data
            print("No faulty data for " + fault_mode + " found in the database for operating condition oc: " + str(oc) + " snr: " + str(snr) + " sev: " + str(sev))
            return None, None

    # Frequency domain data are multidimensional and needs to be treated differently
    if "fft" in filter_to_use.keys():
        healthy_data = pd.DataFrame([d["fft"] for d in healthy_data])
        for key, faulty_test_data in faulty_test_datasets.items():
            faulty_test_datasets[key] = pd.DataFrame([d["fft"] for d in faulty_test_data])
    else:
        healthy_data = pd.DataFrame(healthy_data)
        for key, faulty_test_data in faulty_test_datasets.items():
            faulty_test_datasets[key] = pd.DataFrame(faulty_test_data)

    return healthy_data, faulty_test_datasets


def get_data_from_db_and_save_to_file(db_name):
    # Pull the data from the database

    engineering_features_to_use = [
        'rms',
        # 'sra',
        'kurtosis',
        'crest_factor',
        # 'skewness',
        'ball_h1',
        'ball_h2',
        # 'ball_h3',
        'ball_h4',
        # 'ball_h5',
        'outer_h1',
        'outer_h2',
        # 'outer_h3',
        'outer_h4',
        # 'outer_h5',
        'inner_h1',
        'inner_h2',
        # 'inner_h3',
        'inner_h4',
        # 'inner_h5',
        'entropy',
        'spectral_entropy'
    ]
    # Create the filter that will extract the engineering feature fields from the database
    engineering_filter = {key: 1 for key in
              engineering_features_to_use}  # The filter that will extract only the relevant fields from the database
    engineering_filter.update({"_id": 0})  # Don't extract the id field

    # Create the filter that will extract the frequency feature fields from the database
    fft_filter = {"fft": 1, "_id": 0}

    client = MongoClient()
    db = client[db_name]
    collection = db["raw"]

    prod = product(collection.distinct("oc"), collection.distinct("severity"), collection.distinct("snr"))
    for oc, sev, snr in prod:
        # Get the healthy data
        for filter_to_use, data_type in zip([engineering_filter, fft_filter], ["engineering", "frequency"]):
            healthy, faulty_data_dict = get_data_from_db(oc, sev, snr, db_name, filter_to_use)
            if healthy is not None:
                print("for OC: {} SNR: {} SEV: {}, lengths are: healthy {}, faulty {}".format(oc, snr, sev, len(healthy), [len(faulty) for faulty in faulty_data_dict.values()]))

                 # Calculate the ground truth fault direction
                ground_truth_fault_dir = pd.concat(list(faulty_data_dict.values()), ignore_index=True).mean()-healthy.mean()

                name = db_name + '_{}_oc{}_snr{}_sev{}'.format(data_type,oc, snr, sev)
                metadata = {'ground_truth_fault_direction': list(ground_truth_fault_dir),
                            'dataset_name': name,
                            "snr": snr,
                            }


                export_data_to_file_structure(dataset_name=name,
                                              healthy_data=healthy,
                                              faulty_data_dict=faulty_data_dict,
                                              export_path=pathlib.Path(
                                                  "/home/douwm/projects/PhD/code/biased_anomaly_detection/data"),
                                                metadata=metadata
                                              )


db_name = "lms"
get_data_from_db_and_save_to_file(db_name)