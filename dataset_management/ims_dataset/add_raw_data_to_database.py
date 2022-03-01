# Usefull snippets from kaggle page https://www.kaggle.com/furkancitil/nasa-bearing-dataset-rul-predictionk
import numpy as np
import pandas as pd
import os

from file_definitions import ims_path

folders_for_different_tests = ["1st_test", "2nd_test", "3rd_test/txt"]  # Each folder has text files with the data

folder = folders_for_different_tests[2]


class IMSTest(object):
    def __init__(self, folder_name, channel_info):
        self.folder_name = folder_name
        self.channel_info = channel_info
        self.channel_names = [channel_dict["measurement_name"] for channel_dict in self.channel_info]  # Extract the names of the channels

        self.number_of_measurements = len(list(ims_path.joinpath(self.folder_name).glob('**/*')))
        self.measurement_paths = list(ims_path.joinpath(self.folder_name).glob('**/*'))[0:100]

        self.n_samples_per_measurement = 20480
        self.data_per_channel = [np.zeros([self.number_of_measurements, self.n_samples_per_measurement]) for channel in channel_info] # Create an empty array for each dataset

    def read_files(self):
        # for measurement_id, filepath in enumerate(list(self.measurement_paths)[0:10]):  # For developement
        for measurement_id, filepath in enumerate(self.measurement_paths):  # Loop through all files in the directory and read the data as pd dataframe
            measurement = pd.read_csv(filepath, sep="\t", names=self.channel_names)

            if len(measurement.index) != self.n_samples_per_measurement:
                raise IndexError("The number of samples in the file is different than expected: ".format(len(measurement.index)))

            for channel_id, channel_name in enumerate(measurement.columns):
                measurement_for_channel = measurement[channel_name].values
                self.data_per_channel[channel_id][measurement_id] = measurement_for_channel


        print(measurement_for_channel)


# print(measurement)
# df = pd.read_csv(ims_path.joinpath(folder))

channel_info_test_1 = [
    {
        "measurement_name": "bearing1_channel1",
        "mode": None
    },
    {
        "measurement_name": "bearing1_channel2",
        "mode": None
    },

    {
        "measurement_name": "bearing2_channel1",
        "mode": None
    },
    {
        "measurement_name": "bearing2_channel2",
        "mode": None
    },

    {
        "measurement_name": "bearing3_channel1",
        "mode": "inner"
    },
    {
        "measurement_name": "bearing3_channel2",
        "mode": "inner"
    },

    {
        "measurement_name": "bearing4_channel1",
        "mode": "ball"
    },
    {
        "measurement_name": "bearing4_channel2",
        "mode": "ball"
    },
]

test1 = IMSTest(folders_for_different_tests[0],channel_info_test_1)
test1.read_files()

print(test1.data_per_channel[-1])

