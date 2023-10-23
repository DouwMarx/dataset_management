import itertools
import pathlib
import pickle
import random
from datetime import datetime
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from dataset_management.ultils.write_data_in_standard_format import export_data_to_file_structure
from file_definitions import ims_path
from pypm.phenomenological_bearing_model.bearing_model import Bearing
from tqdm import tqdm


class IMSTest(object):
    """
    Used to add the IMS data to mongodb
    """

    def __init__(self, path_to_measurement_campaign : pathlib.Path, channel_info, rapid_level = None):
        self.folder_name = path_to_measurement_campaign.name # The folder where the text files are stored

        self.channel_info = channel_info  # Info defined for the channel
        self.channel_names = [channel_dict["measurement_name"] for channel_dict in self.channel_info]  # Extract the names of the channels
        self.rotation_frequency = 2000 / 60  # rev/s  , Hz # From the IMS document

        self.rapid_level = rapid_level # Reduce size for prototyping purposes

        self.measurement_paths = list(path_to_measurement_campaign.iterdir())

        # Make sure the record numbers follow the correct time stamps for different samples.
        self.time_stamps = sorted(
            [datetime.strptime(path.name, '%Y.%m.%d.%H.%M.%S') for path in self.measurement_paths])
        self.record_numbers = {self.time_stamps[i]: str(i + 1) for i in range(len(self.time_stamps))} # Notice 1-based indexing for record number

        self.n_records = len(self.measurement_paths)

        self.n_samples_per_measurement = 20480

        d = 8.4
        D = 71.5
        n_ball = 16
        contact_angle = 0.2647664475275398

        self.bearing_geom_obj = Bearing(d, D, contact_angle, n_ball)

        self.ims_meta_data = {'d': d,
                              'D': D,
                              'n_ball': n_ball,
                              'contact_angle': contact_angle,
                              'sampling_frequency': 20480,
                              # Notice that the sampling frequency here is not 20kHz as mentioned in the IMS document but slightly different as suggested by Liu and Gryllias 2020.
                              "expected_fault_frequencies": {
                                  fault_type: self.bearing_geom_obj.get_expected_fault_frequency(fault_type,
                                                                                                 self.rotation_frequency)
                                  for fault_type in ["ball", "outer", "inner"]}
                              }

    def get_artificial_severity_index(self, record_number, channel):
        """
        This will give a severity between 0 and 10 for the samples based on record numbers.
        It starts numbering (1) after the healthy data with the higest severity being 10.
        :return:
        """

        healthy_start_index = self.channel_info[channel]["healthy_records"][0]
        healthy_end_index = self.channel_info[channel]["healthy_records"][1]
        if record_number <= healthy_start_index:
            return -1  # Before healthy, discarded data

        elif record_number <= healthy_end_index:
            return 0  # Healthy data

        else:
            return int(np.ceil(10 * (record_number - healthy_end_index) / (self.n_records - healthy_end_index)))

    def read_file_as_df(self, file_path):
        measurement = pd.read_csv(file_path, sep="\t", names=self.channel_names)
        if len(measurement.index) != self.n_samples_per_measurement:
            raise IndexError(
                "The number of samples in the file is different than expected: ".format(len(measurement.index)))
        return measurement

    def create_document_per_channel(self, filepath):
        dataframe_of_measurement_for_each_channel = self.read_file_as_df(filepath)

        docs_per_channel = {col: "dum" for col in dataframe_of_measurement_for_each_channel.columns} # Empty list for each channel in the df
        for channel_id, channel_name in enumerate(dataframe_of_measurement_for_each_channel.columns):
            measurement_for_channel = dataframe_of_measurement_for_each_channel[channel_name].values
            doc = self.create_document(list(measurement_for_channel), channel_id, filepath)
            # doc_per_channel.append(doc)
            docs_per_channel[channel_name] = doc
        # return doc_per_channel
        return docs_per_channel

    def create_document(self, time_series_data, channel_id, file_path):
        info_for_channel = self.channel_info[channel_id]

        time_stamp = datetime.strptime(file_path.name, '%Y.%m.%d.%H.%M.%S')
        record_number = self.record_numbers[time_stamp]
        sev_index = self.get_artificial_severity_index(int(record_number), channel_id)

        doc = {"mode": info_for_channel["mode"],
               "severity": sev_index,
               "meta_data": self.ims_meta_data,
               "time_series": list(time_series_data),
               "file_path": file_path.as_posix(),  # Convert to string for mongodb to be able to store it
               "time_stamp": time_stamp,
               "record_number": int(record_number),
               "augmented": False,
               "ims_test_number": self.folder_name[0],  # First string of folder name
               "ims_channel_number": str(int(channel_id + 1))  # IMS convention is 1-based channel numbering
               }
        return doc

    def write_to_file(self, target_location = None):

        if target_location is None:
           target_location = pathlib.Path("/home/douwm/projects/PhD/code/biased_anomaly_detection/data")

        process = self.create_document_per_channel

        # measurement_paths = np.random.choice(self.measurement_paths, size=20, replace=False)
        measurement_paths = self.measurement_paths

        # Serial
        # results = [process(path) for path in tqdm(measurement_paths)]

        # Parallel
        results = Parallel(n_jobs=14)(delayed(process)(path) for path in tqdm(measurement_paths))

        for channel in self.channel_names:
            fault_mode_for_channel =  self.channel_info[self.channel_names.index(channel)]["mode"]

            samples_for_channel = [res_dict[channel] for res_dict in results]
            channel_df = pd.DataFrame(samples_for_channel)

            healthy_df = channel_df[channel_df["severity"] == 0]
            faulty_df = channel_df[channel_df["severity"] > 5] # Severely damaged data

            # Extract the time signals are objects in the dataframe
            healthy_time_signals = np.vstack(healthy_df["time_series"].values)
            faulty_time_signals = np.vstack(faulty_df["time_series"].values)

            healthy_time_signals = pd.DataFrame(healthy_time_signals)
            # Make sure there is a channel dimension such that (batch, channel, time)
            faulty_time_signals = {str(fault_mode_for_channel): np.expand_dims(faulty_time_signals, axis=1)}

            # Get the

            export_data_to_file_structure(dataset_name= "ims" + "_test"+ self.folder_name[0]+ "_" + channel,
                                          healthy_data=healthy_time_signals,
                                          faulty_data_dict=  faulty_time_signals,
                                          export_path= target_location,
                                          metadata= self.ims_meta_data
                                          )

    def serial(self):
        return [self.create_document_per_channel(path) for path in tqdm(self.measurement_paths[0:100])]



if __name__ == "__main__":

    from dataset_management.ims_dataset.experiment_meta_data import channel_info,test_folder_names

    for key, folder in test_folder_names.items():
        folder_channel_info = channel_info[key]
        path_to_folder = ims_path.joinpath(folder)
        test_obj = IMSTest(path_to_folder, folder_channel_info)
        test_obj.write_to_file("ims")
