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
from dataset_management.ultils.mongo_test_train_split import test_train_split


class IMSTest(object):
    """
    Used to add the IMS data to mongodb
    """

    def __init__(self, folder_name, channel_info, rapid_level = None):
        self.folder_name = folder_name  # The folder where the text files are stored
        self.channel_info = channel_info  # Info defined for the channel
        self.channel_names = [channel_dict["measurement_name"] for channel_dict in
                              self.channel_info]  # Extract the names of the channels
        self.rotation_frequency = 2000 / 60  # rev/s  , Hz

        self.rapid_level = rapid_level

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
        It starts numbering (1) after the healthy data with the higest severity being 10.
        :return:
        """

        healthy_start_index = self.channel_info[channel]["healthy_records"][0]
        healthy_end_index = self.channel_info[channel]["healthy_records"][1]
        if record_number < healthy_start_index:
            return -1  # Before healthy, discarded data

        elif record_number < healthy_end_index:
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
               "time_stamp": time_stamp,
               "record_number": int(record_number),
               "augmented": False,
               "ims_test_number": self.folder_name[0],  # First string of folder name
               "ims_channel_number": str(int(channel_id + 1))  # IMS convention is 1-based channel numbering
               }
        return doc

    def add_to_db(self, target_db_root):

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

        if self.rapid_level == "0":
            n_per_batch = 2
            batch_ids = range(0, self.n_records, n_per_batch)
            random.seed(12)
            batch_ids = random.sample(batch_ids,5)

        elif self.rapid_level == "1":
            n_per_batch = 100
            batch_ids = range(0, self.n_records, n_per_batch)
            random.seed(12)
            batch_ids = random.sample(batch_ids, int(len(batch_ids)/4))

        elif self.rapid_level == "2":
            n_per_batch = 100
            batch_ids = range(0, self.n_records, n_per_batch)
            random.seed(12)
            batch_ids = random.sample(batch_ids, int(len(batch_ids) / 2))

            # healthy_start = self.channel_info[0]["healthy_records"][0]
            # healthy_end= self.channel_info[0]["healthy_records"][1]
            # batch_ids = [healthy_start - int(n_per_batch/2),healthy_end - int(n_per_batch/2), self.n_records - int(n_per_batch/2)]
        else:
            n_per_batch = 100
            batch_ids = range(0, self.n_records, n_per_batch)

        # TODO: Add the test functionality here to make it around the healhty damage treshold

        # This ensures that all data is first dropped for a certain dataset (channel) before adding data.
        for channel in self.channel_names:
            db_name = target_db_root+ "_test"+ self.folder_name[0]+ "_" + channel
            db, client = make_db(db_name)
            for name in db.collection_names():
                db.drop_collection(name)
                print("Dumped data for dataset {}, collection {}".format(db_name,name))

        for batch_start in tqdm(batch_ids):
            batch_result = [process(sample_name) for sample_name in
                            self.measurement_paths[batch_start:batch_start + n_per_batch]]
            # batch_result = list(itertools.chain(*batch_result))
            for channel in batch_result[0].keys():
                docs = [res_dict[channel] for res_dict in batch_result]
                db, client = make_db(target_db_root+ "_test"+ self.folder_name[0]+ "_" + channel)
                db["raw"].insert_many(docs)
                client.close()

        # This splits the data into train and test sets. It is done here since separate datasets are created for the same test
        for channel in self.channel_names:
            db_name = target_db_root+ "_test"+ self.folder_name[0]+ "_" + channel
            print("Applying test train split on " +db_name)
            test_train_split(db_name)



    def serial(self):
        return [self.create_document_per_channel(path) for path in tqdm(self.measurement_paths[0:100])]



def main(db_to_act_on):
    from dataset_management.ims_dataset.experiment_meta_data import channel_info
    # db, client = make_db(db_to_act_on)

    # for name in db.collection_names():
    #     db.drop_collection(name)
    #
    # db, client = make_db(db_to_act_on)
    # # First clear out the database
    # db["raw"].delete_many({})
    # print("Dumped existing data")

    # for name in db.collection_names():
    #     db.drop_collection(name)


    if "rapid" in db_to_act_on:
        rapid_level = "0"#db_to_act_on[-1] # Take the rapid number from the db name 0 is very rapid, 1 is mildly rapid

        db_to_act_on = "ims_rapid" + rapid_level # Notice that db to act on is rewritten here because the dataset is split automatically by the raw procesing
        print("running rapid")
    else:
        db_to_act_on = "ims" # Notice that db to act on is rewritten here because the dataset is split automatically by the raw procesing
        rapid_level = None

    folders_for_different_tests = ["1st_test", "2nd_test", "3rd_test/txt"]  # Each folder has text files with the data

    for folder, channel_info in zip(folders_for_different_tests, channel_info):
        test_obj = IMSTest(folder, channel_info, rapid_level = rapid_level)
        test_obj.add_to_db(db_to_act_on)
        print("Done with one folder")
    return "nothing"


if __name__ == "__main__":
    # main("ims")
    # main("ims_rapid1")
    main("ims")
