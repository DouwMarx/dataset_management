import pathlib
from itertools import product

import numpy as np
import pandas as pd
from pymongo import MongoClient

from dataset_management.ultils.gradient_prescription import TriangularPeaks
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
        print("No healthy data found in the database for operating condition oc: " + str(
            oc) + " snr: " + str(snr) + " sev: " + str(sev))
        return None, None,None
    else:
        print("Found {} healthy data points".format(len(healthy_data)))

    # Find the corresponding faulty data for each fault mode
    faulty_test_datasets = {}
    all_modes = collection.distinct("mode", filter={"oc": oc, "severity": sev, "snr": snr})  # This includes healthy
    faulty_modes = [mode for mode in all_modes if mode != "healthy"]
    for fault_mode in faulty_modes:
        # Extract the faulty data for a given fault mode
        faulty_test_datasets[fault_mode] = list(collection.find(
            {"oc": oc,
             "severity": sev,
             "snr": snr,
             "mode": fault_mode,
             }
            , filter_to_use)
        )
        # If there is no data for any mode then skip this operating condition
        if sum([len(val) for key,val in faulty_test_datasets.items()]) == 0:
            print("No faulty data for any fault mode found in the database for operating condition oc: " + str(
                oc) + " snr: " + str(snr) + " sev: " + str(sev))
            return None, None,None

    faulty_test_data_expected_fault_freqs = collection.find_one(
        {"oc": oc,
        "severity": sev,
         "snr": snr,
         "mode":  {"$ne": "healthy"},
         "all_expected_fault_frequencies": {"$exists": True}})["all_expected_fault_frequencies"]

    # Frequency domain data and time series data are multidimensional and needs to be treated differently
    if "fft" in filter_to_use.keys():
        healthy_data = pd.DataFrame([d["fft"] for d in healthy_data])
        for key, faulty_test_data in faulty_test_datasets.items():
            faulty_test_datasets[key] = pd.DataFrame([d["fft"] for d in faulty_test_data])
    elif "time_series" in filter_to_use.keys():
                healthy_data = pd.DataFrame([d["time_series"] for d in healthy_data])
                for key, faulty_test_data in faulty_test_datasets.items():
                    faulty_test_datasets[key] = pd.DataFrame([d["time_series"] for d in faulty_test_data])
    else:
        healthy_data = pd.DataFrame(healthy_data)
        for key, faulty_test_data in faulty_test_datasets.items():
            faulty_test_datasets[key] = pd.DataFrame(faulty_test_data)

    return healthy_data, faulty_test_datasets, faulty_test_data_expected_fault_freqs


def get_data_from_db_and_save_to_file(db_name):
    # Pull engineering features and frequency features from the database and save them to the standard file structure

    engineering_features_to_use = [
        'rms',
        # 'sra',
        'kurtosis',
        'crest_factor',
        # 'skewness',
        'ball_h1',
        'ball_h2',
        'ball_h3',
        'ball_h4',
        # 'ball_h5',
        'outer_h1',
        'outer_h2',
        'outer_h3',
        'outer_h4',
        # 'outer_h5',
        'inner_h1',
        'inner_h2',
        'inner_h3',
        'inner_h4',
        # 'inner_h5',
        'entropy',
        'spectral_entropy'
    ]
    # Create the filter that will extract the engineering feature fields from the database
    engineering_filter = {key: 1 for key in
                          engineering_features_to_use}  # The filter that will extract only the relevant fields from the database
    engineering_filter.update({"_id": 0})  # Don't extract the id field

    # Create the filters that will extract frequency fields and  time series respectively from the database
    fft_filter = {"fft": 1, "_id": 0}
    time_series_filter = {"time_series": 1, "_id": 0}

    client = MongoClient()
    db = client[db_name]
    collection = db["raw"]
    db_metadata = db["meta_data"].find_one({"_id": "meta_data"})

    freqs = np.fft.fftfreq(db_metadata["cut_signal_length"], d=1/db_metadata["sampling_frequency"])
    freqs = freqs[1:db_metadata["cut_signal_length"] // 4] # Only use half of the the positive frequencies and remove the DC component
    peak_simulator = TriangularPeaks(freqs)

    severites = collection.distinct("severity")
    severity_excluding_healthy = [sev for sev in severites if sev != 0]
    prod = product(collection.distinct("oc"), severity_excluding_healthy, collection.distinct("snr"))
    for oc, sev, snr in prod:
        # Get the healthy data
        # for filter_to_use, data_type in zip([engineering_filter, fft_filter,time_series_filter], ["engineering", "frequency","time_series"]):
        for filter_to_use, data_type in zip([time_series_filter],["time_series"]):
            healthy, faulty_data_dict,faulty_test_data_expected_fault_freqs = get_data_from_db(oc, sev, snr, db_name, filter_to_use)

            if healthy is not None:
                print(
                    "for OC: {} SNR: {} SEV: {}, lengths are: healthy {}, faulty {}".format(oc, snr, sev, len(healthy),
                                                                                            [len(faulty) for faulty in
                                                                                             faulty_data_dict.values()])
                )
                meta_data = {}

                # Calculate the ground truth fault direction
                ground_truth_fault_dir = pd.concat(list(faulty_data_dict.values()),
                                                   ignore_index=True).mean() - healthy.mean()

                if data_type == "engineering":
                    meta_data.update({"expected_fault_direction": list(np.ones(len(ground_truth_fault_dir)))}) # Assumption that all engineering features are equally important and are increasing with fault severity
                elif data_type == "frequency":
                    cumulative_expected_fault_direction = np.zeros(len(ground_truth_fault_dir))
                    for mode, expected_fault_freq in faulty_test_data_expected_fault_freqs.items():
                        expected_fault_direction_for_mode = peak_simulator.get_expected_fault_behaviour(1,expected_fault_freq)
                        meta_data.update({"expected_fault_direction_" + mode: list(expected_fault_direction_for_mode)})
                        cumulative_expected_fault_direction += expected_fault_direction_for_mode
                    meta_data.update({"expected_fault_direction": list(cumulative_expected_fault_direction)})


                name = db_name + '_{}_oc{}_snr{}_sev{}'.format(data_type, oc, snr, sev)
                meta_data.update({
                    'ground_truth_fault_direction': list(ground_truth_fault_dir),
                    'dataset_name': name,
                    "snr": snr,
                    "severity": sev,
                })

                meta_data.update(db_metadata)

                export_data_to_file_structure(dataset_name=name,
                                              healthy_data=healthy,
                                              faulty_data_dict=faulty_data_dict,
                                              export_path=pathlib.Path(
                                                  "/home/douwm/projects/PhD/code/biased_anomaly_detection/data"),
                                              metadata=meta_data
                                              )

                # Also export the time domain data directly to the file structure



def main(db_name):
    get_data_from_db_and_save_to_file(db_name)

if __name__ == '__main__':
    main("cwr")
