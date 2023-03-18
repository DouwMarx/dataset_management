import random
from datetime import datetime
import numpy as np
import pandas as pd
from database_definitions import make_db
from file_definitions import ims_path
from pypm.phenomenological_bearing_model.bearing_model import Bearing
from tqdm import tqdm
from dataset_management.ultils.mongo_test_train_split import test_train_split
from scipy.io import loadmat
from file_definitions import lms_path
import matplotlib.pyplot as plt


def overlap(array, len_chunk, len_sep=1):
    """Returns a matrix of all full overlapping chunks of the input `array`, with a chunk
    length of `len_chunk` and a separation length of `len_sep`. Begins with the first full
    chunk in the array.

     from https://stackoverflow.com/questions/38163366/split-list-into-separate-but-overlapping-chunks
     """

    n_arrays = np.int(np.ceil((array.size - len_chunk + 1) / len_sep))

    array_matrix = np.tile(array, n_arrays).reshape(n_arrays, -1)

    columns = np.array(((len_sep * np.arange(0, n_arrays)).reshape(n_arrays, -1) + np.tile(
        np.arange(0, len_chunk), n_arrays).reshape(n_arrays, -1)), dtype=np.intp)

    rows = np.array((np.arange(n_arrays).reshape(n_arrays, -1) + np.tile(
        np.zeros(len_chunk), n_arrays).reshape(n_arrays, -1)), dtype=np.intp)

    return array_matrix[rows, columns]


class LMS(object):
    """
    Used to add the IMS data to mongodb
    """

    def __init__(self):
        self.datasets = {
            "SOR2.mat": {"mode": "outer",
                         "oc": 2,
                         "severity": 1},
            "SOR1.mat": {"mode": "outer",
                         "oc": 1,
                         "severity": 1},
            "MIR3.mat": {"mode": "inner",
                         "oc": 3,
                         "severity": 2},
            "MIR2.mat": {"mode": "inner",
                         "oc": 2,
                         "severity": 2},
            "MIR1.mat": {"mode": "inner",
                         "oc": 1,
                         "severity": 2},
            "LOR3.mat": {"mode": "outer",
                         "oc": 3,
                         "severity": 2},
            "LOR2.mat": {"mode": "outer",
                         "oc": 2,
                         "severity": 2},
            "LOR1.mat": {"mode": "outer",
                         "oc": 1,
                         "severity": 2},
            "HEA2.mat": {"mode": "healthy",
                         "oc": 2,
                         "severity": 0},
            "HEA1.mat": {"mode": "healthy",
                         "oc": 1,
                         "severity": 0},

        }

        # The ranges for different speeds as read from the plotted speed signal
        self.speed_ranges = {"4": [int(3.013e6), int(3.423e6)],
                             "3": [int(2.216e6), int(2.704e6)],
                             "2": [int(1.664e6), int(1.958e6)],
                             "1": [int(9.05e5), int(1.261e6)]}

        self.lms_meta_data = {"sampling_frequency": 51200}

        self.sampling_frequency = 51200  # Sampling rate derived from increment parameter in the "x_values" field
        rotation_speed = 20  # rev/s
        # time for 10 revolutions
        t_10revs = 14 / rotation_speed
        # number of samples for 10 revolutions
        n_samples_10revs = t_10revs * self.sampling_frequency
        self.cut_signal_length = int(n_samples_10revs)
        print("Cutting signals in length: ", self.cut_signal_length)

        self.db, self.client = make_db("lms")

        # Drop the previous data
        self.db.drop_collection("raw")
        self.db.drop_collection("meta_data")

        # From SKF website for bearing 2206 EKTN9
        self.freq_per_rpm = {
            # "cage": 0.394,
            "ball": 2.196,
            "inner": 7.276,
            "outer": 4.724
        }

        # Add a document to the db with _id meta_data to store the meta data
        self.db["meta_data"].insert_one({"_id": "meta_data",
                                         "signal_length": self.cut_signal_length,
                                         "sampling_frequency": self.sampling_frequency,
                                         "n_faults_per_revolution": self.freq_per_rpm,
                                         "dataset_name": "CWR",
                                         })


    def get_accelerometer_signal(self, path_to_mat_file):
        mat = loadmat(str(path_to_mat_file), matlab_compatible=True, simplify_cells=True)
        signals = []
        accelerometer_signals = mat["Signal_3"]["y_values"]["values"]  # Signal_3 is the accelerometer
        speed_signal = mat["Signal_1"]["y_values"]["values"]  # Signal_1 is the speed signal
        speed_correction_factor = mat["Signal_1"]["y_values"]["quantity"]["unit_transformation"]["factor"]

        if "IR" in path_to_mat_file.name:
            freq = self.freq_per_rpm["inner"]
        elif "OR" in path_to_mat_file.name:
            freq = self.freq_per_rpm["outer"]
        else:
            freq = np.nan

        for accelerometer in [0, 1]:
            sig = accelerometer_signals[:, accelerometer]  # Numbering is not necessarily correct
            for speed, samples in self.speed_ranges.items(): # TODO: Need to verify that speed ranges are correct
                speed_samples = speed_signal[samples[0]:samples[1]]*speed_correction_factor # In RPM
                average_speed = np.mean(speed_samples)
                speed_fluctuation = np.std(speed_samples)
                print("average speed: ", average_speed, "RPM", "speed fluctuation: ", speed_fluctuation, "RPM" )
                signals.append({"signal": sig[samples[0]:samples[1]],
                                "rpm": average_speed,
                                "rpm_fluctuation": speed_fluctuation,
                                "accelerometer": accelerometer,
                                "expected_fault_frequency": freq*average_speed/60,
                                })
        return signals

    def create_document(self, time_series_data, fault_class, severity, operating_condition, accelerometer_number,
                        speed,expected_fault_frequency):
        doc = {"mode": fault_class,
               "severity": severity,
               "meta_data": self.lms_meta_data,
               "time_series": list(time_series_data),
               "oc": operating_condition,
               "accelerometer_number": accelerometer_number,
               "rpm": speed,
               "snr": 0,
               "sampling_frequency": self.lms_meta_data["sampling_frequency"],
                "expected_fault_frequency": expected_fault_frequency
               }
        return doc

    def add_to_db(self, signal_segments, mode, severity, operating_condition, accelerometer_number, speed, expected_fault_frequency):
        docs = [self.create_document(signal, mode, severity, operating_condition, accelerometer_number, speed,expected_fault_frequency) for
                signal in
                signal_segments]

        # TODO: Add the test functionality here to make it around the healthy damage treshold
        self.db["raw"].insert_many(docs)

    def add_all_to_db(self):
        for key, val in tqdm(self.datasets.items()):  # TODO: Notice use of last 10 seconds of data
            signals = self.get_accelerometer_signal(lms_path.joinpath(key))

            for signal in signals:
                signal_segments = overlap(signal["signal"], self.cut_signal_length, int(self.cut_signal_length / 8))
                self.add_to_db(signal_segments, val["mode"], val["severity"], val["oc"], signal["accelerometer"],
                               signal["rpm"],signal["expected_fault_frequency"])


o = LMS()

o.add_all_to_db()

# Accelerometer data is "Signal_3", There are two channels for signal 3, one for each of the accelerometers
# This is used to explore the data channels

# example = o.datasets.items().__iter__().__next__()
# mat = loadmat(lms_path.joinpath(example[0]), matlab_compatible=True, simplify_cells=True)

# # For channel
# for key, val in mat.items():
#     if "Signal" in key:
#         print("")
#         print(key, val["function_record"]["name"])
#
# # For sampling rate
# print("")
# increment = mat["Signal_1"]["x_values"]["increment"]
# print("'Increment':", increment)
# print("1/increment = fs:", 1 / increment)
#
# # Plot the three tacho signals
# # plt.figure()
# # for i in range(1,4):
# #     plt.plot(mat["Signal_{}".format(i)]["y_values"]["values"],label="Signal_{}".format(i))
# # plt.legend()
# # Signal 1 is the processed tacho that is likely in RPM
#
# plt.figure()
# sigs = mat["Signal_3"]["y_values"]["values"][int(2e6):int(3e6)]
# for name,sig in enumerate(sigs.T):
#     plt.plot(sig, label=name)
#     plt.legend()


#
# # Sample ranges of the different speeds read from the graph
# ranges = {"4": [int(3.013e6), int(3.423e6)],
#           "3": [int(2.216e6), int(2.704e6)],
#           "2": [int(1.664e6), int(1.958e6)],
#           "1": [int(9.05e5), int(1.261e6)]}
