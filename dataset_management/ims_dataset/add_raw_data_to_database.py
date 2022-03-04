# Usefull snippets from kaggle page https://www.kaggle.com/furkancitil/nasa-bearing-dataset-rul-predictionk
import itertools
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from experiment_meta_data import channel_info
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

        if rapid_for_test:
            self.measurement_paths = list(ims_path.joinpath(self.folder_name).iterdir())[
                                     0:10]  # Only load a few of the samples when testing
        else:
            # self.measurement_paths = list(ims_path.joinpath(self.folder_name).glob('**/*'))
            self.measurement_paths = list(ims_path.joinpath(self.folder_name).iterdir())

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
                              'sampling_frequency': 20000,
                              "expected_fault_frequencies":{fault_type:self.bearing_geom_obj.get_expected_fault_frequency(fault_type,self.rotation_frequency) for fault_type in ["ball","outer","inner"]}
                              }

        self.meta_data_for_fault = {}
        for mode in ["ball", "inner", "outer"]:
            geometry_factor_for_mode = self.bearing_geom_obj.get_geometry_parameter(mode)
            average_fault_freq = self.rotation_frequency * geometry_factor_for_mode / (2 * np.pi)

            derived_params = {
                'geometry_factor': geometry_factor_for_mode,
                'average_fault_frequency': average_fault_freq
            }
            self.meta_data_for_fault.update({mode: derived_params})

    def fault_meta_data(self, channel_id):
        """
        Bearing geometry for the Rexnord ZA-2115 bearing is is from the following paper:

        https: // www.researchgate.net / publication / 335937388_Health_Indicator_Construction_Based_on_MD - CUMSUM_with_Multi - Domain_Features_Selection_for_Rolling_Element_Bearing_Fault_Diagnosis

        """
        mode_for_channel = self.channel_info[channel_id]["mode"]

        if mode_for_channel is not None:
            return self.meta_data_for_fault[mode_for_channel]
        else:
            return None

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

        fault_meta_data = self.fault_meta_data(channel_id)

        doc = {"mode": info_for_channel["mode"],
               "severity": str(file_path.stem),  # str(severity), TODO: This should be related to the record number
               "meta_data": fault_meta_data,
               "time_series": list(time_series_data),
               # "time_series": list(time_series_data),
               # "time_series": time_series_data,
               "augmented": False,
               "ims_test_number": self.folder_name[0],  # First string of folder name
               "ims_channel_number": str(int(channel_id + 1))  # IMS convention is 1-based channel numbering
               }
        return doc

    def add_to_db(self):

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

        n_per_batch= 200
        for batch_start in tqdm(range(0, len(self.measurement_paths), n_per_batch)):
            batch_result =  [process(sample_name) for sample_name in self.measurement_paths[batch_start:batch_start + n_per_batch]]
            batch_result = list(itertools.chain(*batch_result))
            db, client = make_db("ims")
            db["raw"].insert_many(batch_result)
            client.close()


    def serial(self):
        return [self.create_document_per_channel(path) for path in tqdm(self.measurement_paths[0:100])]


# First clear out the database
db, client = make_db("ims_test")
db["raw"].delete_many({})
print("Dumped existing data")

folders_for_different_tests = ["1st_test", "2nd_test", "3rd_test/txt"]  # Each folder has text files with the data

for folder, channel_info in zip(folders_for_different_tests, channel_info):
    test_obj = IMSTest(folder, channel_info,rapid_for_test=True)
    test_obj.add_to_db()
    print("Done with one folder")
