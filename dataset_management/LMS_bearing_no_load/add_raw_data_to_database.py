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


def get_accelerometer_signal(path_to_mat_file):
    mat = loadmat(str(path_to_mat_file), matlab_compatible=True, simplify_cells=True)
    accelerometer_signals = mat["Signal_3"]["y_values"]["values"]
    accelerometer_1 = accelerometer_signals[:,0] # Numbering is not necessarily correct
    return accelerometer_1


class LMS(object):
    """
    Used to add the IMS data to mongodb
    """

    def __init__(self):

        self.datasets = {
            "SOR2.mat":{"mode":"outer",
                                     "oc":2,
                                     "severity":1},
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
            "HEA2.mat": {"mode": "health",
                         "oc": 2,
                         "severity": 0},
            "HEA1.mat": {"mode": "health",
                         "oc": 1,
                         "severity": 0},

        }

        self.lms_meta_data = {"sampling_frequency":51200}

        p = lms_path.joinpath("MIR2.mat")
        signal = get_accelerometer_signal(p)
        signal_length = len(signal)

        fs = 51200  # Sampling rate derived from increment parameter in the "x_values" field
        rotation_speed = 20  # rev/s
        # time for 10 revolutions
        t_10revs = 10 / rotation_speed
        # number of samples for 10 revolutions
        n_samples_10revs = t_10revs * fs
        self.cut_signal_length = int(n_samples_10revs)
        print("Cutting signals in length: ", self.cut_signal_length)

        # signal_segments = signal[0:signal_length-signal_length%cut_signal_length].reshape(-1,cut_signal_length)
        # self.signal_segments = overlap(signal, cut_signal_length, int(cut_signal_length / 2))

        self.db,self.client = make_db("lms")
        self.db.drop_collection("raw")


    def create_document(self, time_series_data, fault_class, severity,operating_condition):
        doc = {"mode": fault_class,
               "severity": severity,
               "meta_data": self.lms_meta_data,
               "time_series": list(time_series_data),
               "oc":operating_condition
               }
        return doc

    def add_to_db(self,signal_segments,mode,severity,operating_condition):
        docs = [self.create_document(signal,mode,severity,operating_condition) for signal in signal_segments]

        # TODO: Add the test functionality here to make it around the healhty damage treshold
        self.db["raw"].insert_many(docs)

    def add_all_to_db(self):
        for key,val in tqdm(self.datasets.items()):
            signal = get_accelerometer_signal(lms_path.joinpath(key))[-51200*10:] # Use the last 10 seconds of data for each trial
            print(signal.shape)
            signal_segments = overlap(signal, self.cut_signal_length, int(self.cut_signal_length / 8))
            self.add_to_db(signal_segments,val["mode"],val["severity"],val["oc"])

o = LMS()
# o.add_all_to_db()

# Accelerometer data is "Signal_3", There are two channels for signal 3, one for each of the accelerometers

# This is used to explore the data channels

# example = o.datasets.items().__iter__().__next__()
# signal = get_accelerometer_signal(lms_path.joinpath(example[0]))
#     print(signal.shape)
#     signal_segments = overlap(signal, self.cut_signal_length, int(self.cut_signal_length / 8))
#     self.add_to_db(signal_segments, val["mode"], val["severity"], val["oc"])
# for key, val in l.items():
#     if "Signal" in key:
#         print("")
#         print(key, val["function_record"]["name"])
