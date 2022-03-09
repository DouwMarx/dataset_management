# Usefull snippets from kaggle page https://www.kaggle.com/furkancitil/nasa-bearing-dataset-rul-predictionk
import itertools
import pickle
import random
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from database_definitions import make_db
from file_definitions import ims_path
from pypm.phenomenological_bearing_model.bearing_model import Bearing
from multiprocessing import Pool
from tqdm import tqdm
from time import time


class IMSTest(object):
    """
    Used to add the IMS data to mongodb
    """

    def __init__(self, folder_name, channel_info, rapid_for_test=False):
        self.folder_name = folder_name  # The folder where the text files are stored
        self.channel_info = channel_info  # Info defined for the channel
        self.channel_names = [channel_dict["measurement_name"] for channel_dict in
                              self.channel_info]  # Extract the names of the channels
        # self.number_of_measurements = len(list(ims_path.joinpath(self.folder_name).glob('**/*')))
        # self.number_of_measurements = len(list(ims_path.joinpath(self.folder_name).iterdir()))
        self.rotation_frequency = 2000 / 60  # rev/s  , Hz

        self.rapid_for_test = rapid_for_test

        self.measurement_paths = list(ims_path.joinpath(self.folder_name).iterdir())

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
        It starts numbering after the healthy data
        :return:
        """

        healthy_start_index = self.channel_info[channel]["healthy_records"][0]
        healthy_end_index = self.channel_info[channel]["healthy_records"][1]
        if record_number < healthy_start_index:
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

        doc_per_channel = []
        for channel_id, channel_name in enumerate(dataframe_of_measurement_for_each_channel.columns):
            measurement_for_channel = dataframe_of_measurement_for_each_channel[channel_name].values
            doc = self.create_document(list(measurement_for_channel), channel_id, filepath)
            doc_per_channel.append(doc)
        return doc_per_channel

    def create_document(self, time_series_data, channel_id, file_path):
        info_for_channel = self.channel_info[channel_id]

        # print("test",self.folder_name[0])
        # print("chan",str(int(channel_id + 1)))

        time_stamp = datetime.strptime(file_path.name, '%Y.%m.%d.%H.%M.%S')
        record_number = self.record_numbers[time_stamp]
        sev_index = self.get_artificial_severity_index(int(record_number), channel_id)

        doc = {"mode": info_for_channel["mode"],
               "severity": sev_index,
               "meta_data": self.ims_meta_data,
               "time_series": list(time_series_data),
               "time_stamp": time_stamp,
               "record_number": record_number,
               "augmented": False,
               "ims_test_number": self.folder_name[0],  # First string of folder name
               "ims_channel_number": str(int(channel_id + 1))  # IMS convention is 1-based channel numbering
               }
        return doc

    def add_to_db(self, target_db):

        # Add chucks of documents to the db per time to keep memory manageable

        process = self.create_document_per_channel

        # n_per_run = 280
        # for i in tqdm(range(0, len(self.measurement_paths) + n_per_run, n_per_run)):
        #     result = Parallel(n_jobs=14)(
        #         delayed(process)(i) for i in self.measurement_paths[i:i + n_per_run])
        #     # flattened = itertools.chain.from_iterable(docs)
        #     result = [item for sublist in result for item in sublist]
        #
        #     db, client = make_db("ims")
        #     db["raw"].insert_many(result)
        #     client.close()

        if self.rapid_for_test:
            n_per_batch = 2
            batch_ids = range(0, self.n_records, n_per_batch)
            random.seed(12)
            batch_ids = random.sample(batch_ids,5)

            # healthy_start = self.channel_info[0]["healthy_records"][0]
            # healthy_end= self.channel_info[0]["healthy_records"][1]
            # batch_ids = [healthy_start - int(n_per_batch/2),healthy_end - int(n_per_batch/2), self.n_records - int(n_per_batch/2)]
            print(batch_ids)
        else:
            n_per_batch = 1000
            batch_ids = range(0, self.n_records, n_per_batch)

        # TODO: Add the test functionality here to make it around the healhty damage treshold
        for batch_start in tqdm(batch_ids):
            batch_result = [process(sample_name) for sample_name in
                            self.measurement_paths[batch_start:batch_start + n_per_batch]]
            batch_result = list(itertools.chain(*batch_result))
            db, client = make_db(target_db)
            db["raw"].insert_many(batch_result)
            client.close()

    def serial(self):
        return [self.create_document_per_channel(path) for path in tqdm(self.measurement_paths[0:100])]



def main(db_to_act_on):
    from experiment_meta_data import channel_info
    db, client = make_db(db_to_act_on)
    # First clear out the database
    db["raw"].delete_many({})
    print("Dumped existing data")

    folders_for_different_tests = ["1st_test", "2nd_test", "3rd_test/txt"]  # Each folder has text files with the data

    for folder, channel_info in zip(folders_for_different_tests, channel_info):
        test_obj = IMSTest(folder, channel_info)
        test_obj.add_to_db("ims_test")
        print("Done with one folder")
    return db


def test():
    db_to_act_on = "ims_test"
    from dataset_management.ims_dataset.experiment_meta_data import channel_info
    db, client = make_db(db_to_act_on)
    db["raw"].delete_many({})
    print("Dumped existing data")

    folders_for_different_tests = ["1st_test", "2nd_test", "3rd_test/txt"]  # Each folder has text files with the data

    for folder, channel_info in zip(folders_for_different_tests, channel_info):
        test_obj = IMSTest(folder, channel_info, rapid_for_test=True)
        test_obj.add_to_db("ims_test")
        print("Done with one folder")
    return db

if __name__ == "__main__":
    main("ims")
