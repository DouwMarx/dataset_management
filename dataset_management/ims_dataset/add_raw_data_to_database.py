# Usefull snippets from kaggle page https://www.kaggle.com/furkancitil/nasa-bearing-dataset-rul-predictionk
import pickle

import numpy as np
import pandas as pd
import os

from file_definitions import ims_path

from database_definitions import db_ims

folders_for_different_tests = ["1st_test", "2nd_test", "3rd_test/txt"]  # Each folder has text files with the data

folder = folders_for_different_tests[2]


class IMSTest(object):
    def __init__(self, folder_name, channel_info, n_sev = 10):


        self.folder_name = folder_name
        self.channel_info = channel_info
        self.channel_names = [channel_dict["measurement_name"] for channel_dict in self.channel_info]  # Extract the names of the channels

        self.n_sev = n_sev

        self.number_of_measurements = len(list(ims_path.joinpath(self.folder_name).glob('**/*')))
        self.measurement_paths = list(ims_path.joinpath(self.folder_name).glob('**/*'))[0:10]

        self.n_samples_per_measurement = 20480
        self.data_per_channel = [np.zeros([self.number_of_measurements, self.n_samples_per_measurement]) for channel in channel_info] # Create an empty array for each dataset

        bearing_geometry = {'d': 8.4,
        'D': 71.5,
        'n_ball': 16,
        'contact_angle': 0.2740166925631097,
        'sampling_frequency': 10000,
        't_duration': 1,
        'n_measurements': 100,
        'speed_profile_type': 'constant',
        'derived': {'geometry_factor': 4.201512427465972,
                    'average_fault_frequency': 35.01260356221636},
        'measured_time': [1]}

        self.meta_data = bearing_geometry

        self.read_files() # Read the files into memory

    def read_files(self):
        # for measurement_id, filepath in enumerate(list(self.measurement_paths)[0:10]):  # For developement
        for measurement_id, filepath in enumerate(self.measurement_paths):  # Loop through all files in the directory and read the data as pd dataframe
            measurement = pd.read_csv(filepath, sep="\t", names=self.channel_names)

            if len(measurement.index) != self.n_samples_per_measurement:
                raise IndexError("The number of samples in the file is different than expected: ".format(len(measurement.index)))

            for channel_id, channel_name in enumerate(measurement.columns):
                measurement_for_channel = measurement[channel_name].values
                self.data_per_channel[channel_id][measurement_id] = measurement_for_channel

    def split_channel_into_severities(self, healthy_records, channel_id):
        data_for_channel = self.data_per_channel[channel_id]

        healthy_start = healthy_records[0]
        healthy_end = healthy_records[1]

        sev_groups_healthy = np.array([healthy_start])
        sev_groups_damaged = np.linspace(healthy_end, self.number_of_measurements, self.n_sev, dtype=int,endpoint=False)
        sev_groups = np.hstack([sev_groups_healthy,sev_groups_damaged])
        print(sev_groups)

        data_for_severities = np.split(data_for_channel, sev_groups) # split data into groups with similar severity
        data_for_severities = data_for_severities[1:] # discard any data before the healthy region (run in)

        return data_for_severities

    def create_documents_for_channel(self, channel_id):
        info_for_channel = self.channel_info[channel_id]
        healthy_records = info_for_channel["healthy_records"]

        data_for_severities = self.split_channel_into_severities(healthy_records, channel_id)

        docs_for_channel = []
        for severity,time_series_data in enumerate(data_for_severities):
            doc = {"mode": info_for_channel["mode"] ,
                   "severity": str(severity),
                   "meta_data": self.meta_data,
                   "time_series": pickle.dumps(time_series_data),
                   "augmented": False,
                   "ims_test_number":self.folder_name[0]
                   }
            docs_for_channel.append(doc)

        return docs_for_channel

    def add_to_database(self):
        for id, channel in enumerate(self.channel_names):
            docs_for_channel = self.create_documents_for_channel(id)
            db_ims["raw"].insert_many(docs_for_channel)


        # raw.insert_one(doc)  # Insert document into the collection


# print(measurement)
# df = pd.read_csv(ims_path.joinpath(folder))

channel_info_test_1 = [
    {
        "measurement_name": "bearing1_channel1",
        "mode": None,
        "healthy_records": [10,100]
    },
    {
        "measurement_name": "bearing1_channel2",
        "mode": None,
        "healthy_records": [10, 100]
    },

    {
        "measurement_name": "bearing2_channel1",
        "mode": None,
        "healthy_records": [10, 100]
    },
    {
        "measurement_name": "bearing2_channel2",
        "mode": None,
        "healthy_records": [10, 100]
    },

    {
        "measurement_name": "bearing3_channel1",
        "mode": "inner",
        "healthy_records": [10, 100]
    },
    {
        "measurement_name": "bearing3_channel2",
        "mode": "inner",
        "healthy_records": [10, 100]
    },

    {
        "measurement_name": "bearing4_channel1",
        "mode": "ball",
       "healthy_records": [10, 100]
},
    {
        "measurement_name": "bearing4_channel2",
        "mode": "ball",
        "healthy_records": [10, 100]
    },
]

test1 = IMSTest(folders_for_different_tests[0], channel_info_test_1,n_sev=25)
test1.add_to_database()

