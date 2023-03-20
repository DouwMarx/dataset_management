import numpy as np
import pandas as pd
from pymongo import MongoClient
from scipy.stats import kurtosis

from database_definitions import make_db
from scipy.io import loadmat

from dataset_management.ultils import processing, create_noisy_copies_of_dataset, compute_features
from dataset_management.ultils.time_frequency import get_signal_length_for_number_of_events
from file_definitions import cwr_path


def get_metadata_from_csv():
    meta_data_SmithRandal2014 = pd.read_csv("meta_data_tablea2_Smith-Randal-2014.csv", keep_default_na=False)
    meta_data_SmithRandal2014.replace(np.nan, "0",
                                      inplace=True)  # Replace all NaN with 0 to avoid issues with data types
    return meta_data_SmithRandal2014


class CWR(object):
    """
    Used to add the CWR data to mongodb
    """

    def __init__(self,percentage_overlap = 0.5):

        self.percentage_overlap = percentage_overlap

        rpms = np.array([1797,1772,1750,1730])
        mean_rpm = np.mean(rpms)

        # Use the mean rpm for determining the correct chunk length
        rotation_rate = mean_rpm/ 60  # Rev/s
        self.sampling_frequency = 12000

        min_n_events_per_rev = 2.357
        min_events = 10
        self.cut_signal_length = get_signal_length_for_number_of_events(mean_rpm, min_n_events_per_rev,
                                                                        self.sampling_frequency, min_events)
        print("The cut signal length is: {}".format(self.cut_signal_length))

        self.smith_randal_meta_data = get_metadata_from_csv()

        # First drop all the datasets associated with the cwr dataset
        client = MongoClient()
        db_names = client.list_database_names()  # All the mongo dataset names

        # Drop all the databases associated with the cwr dataset
        for db_name in db_names:
            if "cwr" in db_name:
                client.drop_database(db_name)

        self.db, self.client = make_db("cwr")

        # Add the characteristic number of faults per revolution to the meta data
        self.n_faults_per_revolution = {mode : self.get_expected_fault_frequency_for_mode(mode,60) for mode in ["inner","outer","ball"]}

        # Add a document to the db with _id meta_data to store the meta data
        self.db["meta_data"].insert_one({"_id": "meta_data",
                                         "signal_length": self.cut_signal_length,
                                         "sampling_frequency": self.sampling_frequency,
                                         "n_faults_per_revolution": self.n_faults_per_revolution,
                                         "dataset_name": "CWR",
                                         })

    def get_expected_fault_frequency_for_mode(self, mode, rpm):
        rotation_rate = rpm / 60  # Rev/s
        # Fault frequencies from Smith and Randal 2014
        if "inner" in mode:
            return 5.415 * rotation_rate
        elif "ball" in mode:
            return 2.357 * rotation_rate
        elif "outer" in mode:
            return 3.585 * rotation_rate
        else:
            return None

    def create_document(self, time_series_data, metadata):
        doc = metadata.copy()
        doc["time_series"] = list(time_series_data)
        return doc

    def add_to_db(self, signal_segments, meta_data):
        docs = [self.create_document(signal, meta_data) for signal in signal_segments]

        # TODO: Add the test functionality here to make it around the healthy damage threshold
        self.db["raw"].insert_many(docs)  # Insert the documents into the db with the right operating condition

    def get_meta_data(self, stem):
        file_name_number = int(stem)

        r = self.smith_randal_meta_data

        # Find the row and column position in the dataframe where the number of the meaurement occurs
        row, column = np.where(r == file_name_number)
        fault_width = r.iloc[row, 0].values[0]  # Fault width is in the first column
        hp = r.iloc[row, 1].values[0]  # Power (Operating condition) is in the second column
        rpm = r.iloc[row, 2].values[0]  # Speed is in the third column
        mode = str(r.columns[column].values[0])  # The mode is in the column name
        expected_fault_frequency = self.get_expected_fault_frequency_for_mode(mode, rpm)

        meta_data = {"severity": fault_width,
                     "oc": int(hp),  # Operating condition
                     "snr": 0,  # Signal-to-noise ratio is zero for the raw data
                     "mode": mode, # Note that the mode is sometimes more descriptive like outer orthogonal. Here we only use the "outer centre" for now.
                     "rpm": int(rpm),
                     "expected_fault_frequency": float(
                         expected_fault_frequency) if expected_fault_frequency != None else None,
                     "dataset_number": file_name_number,
                     "sampling_frequency": self.sampling_frequency  # Hz
                     }
        return meta_data

    def add_all_to_db(self):
        # loop through all files in the pathlib path directory
        for file_name in cwr_path.glob("*.mat"):
            meta_data = self.get_meta_data(file_name.stem)

            # Do not add outer data other than outer centre
            # Also, rename the outer centre data to just outer
            if "outer" in meta_data["mode"]:
                if "centre" in meta_data["mode"]:
                    meta_data["mode"] = "outer"
                else:
                    print("Skipping data for mode: ", meta_data["mode"])
                    continue


            path_to_mat_file = cwr_path.joinpath(file_name.name)
            mat = loadmat(str(path_to_mat_file))  # Load the .mat file
            key = [key for key in mat.keys() if "DE" in key][0]  # Here we select the drive-end measurements # TODO: Notice we are using the drive-end measurements
            signal = mat[key].flatten()  # Here we select the drive-end measurements

            if meta_data["severity"] == 0:  # The healthy data is sampled at a higher rate, need to downsample
                signal = signal[
                         ::4].copy()  # Down sample the healthy data since it is sampled at a different sampling rate than the damaged data. 12kHz vs 48kHz
                # TODO: Notice that the down-sampling can also be done with different phase.
                print("Down sampling healthy data, dataset ", meta_data["dataset_number"])
                print("Healthy data mean and std: ", np.mean(signal), np.std(signal))
                print("Meta data: ", meta_data)


            signal_segments = processing.overlap(signal, self.cut_signal_length, self.percentage_overlap )  # Segments have half overlap

            self.add_to_db(signal_segments, meta_data)
        return self.db



print("Adding CWR data to db")
CWR().add_all_to_db()

print("\n \n Creating noisy copies of the data")
create_noisy_copies_of_dataset.main("cwr")

print("\n \n Computing features")
compute_features.main("cwr")
