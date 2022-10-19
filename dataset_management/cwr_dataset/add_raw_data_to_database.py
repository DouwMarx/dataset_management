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
from file_definitions import cwr_path


def overlap(array, len_chunk, len_sep=1):
    """Returns a matrix of all full overlapping chunks of the input `array`, with a chunk
    length of `len_chunk` and a separation length of `len_sep`. Begins with the first full
    chunk in the array.

     from https://stackoverflow.com/questions/38163366/split-list-into-separate-but-overlapping-chunks
     """

    n_arrays = int(np.ceil((array.size - len_chunk + 1) / len_sep))

    array_matrix = np.tile(array, n_arrays).reshape(n_arrays, -1)

    columns = np.array(((len_sep * np.arange(0, n_arrays)).reshape(n_arrays, -1) + np.tile(
        np.arange(0, len_chunk), n_arrays).reshape(n_arrays, -1)), dtype=np.intp)

    rows = np.array((np.arange(n_arrays).reshape(n_arrays, -1) + np.tile(
        np.zeros(len_chunk), n_arrays).reshape(n_arrays, -1)), dtype=np.intp)

    return array_matrix[rows, columns]


def get_accelerometer_signal(path_to_mat_file):
    mat = loadmat(str(path_to_mat_file), matlab_compatible=True, simplify_cells=True)
    accelerometer_signals = mat["Signal_3"]["y_values"]["values"]
    accelerometer_1 = accelerometer_signals[:, 0]  # Numbering is not necessarily correct
    return accelerometer_1


class CWR(object):
    """
    Used to add the CWR data to mongodb
    """

    def __init__(self):
        sampling_frequency = 12000  # Hz
        rotation_rate = 1772 / 60  # Rev/s

        self.files_and_measurement_ids = {
            "210.mat": {"mode": "inner",
                        "expected_fault_frequency": 5.415 * rotation_rate,
                        "severity":1
                        },
            "223.mat": {"mode": "ball",
                        "expected_fault_frequency": 2.357 * rotation_rate,
                        "severity":1
                        },
            "235.mat": {"mode": "outer",
                        "expected_fault_frequency": 3.585 * rotation_rate,
                        "severity":1
                        },
            "098.mat": {"mode": None,
                        "expected_fault_frequency": None,
                        "severity":1
                        },
        }

        # For Ball failure mode having the lowest expected fault frequency
        lowest_expected_fault_frequency = self.files_and_measurement_ids["223.mat"]["expected_fault_frequency"]
        n_events = 15
        # time required for n_events for highest fault frequency
        duration_for_n_events = n_events / lowest_expected_fault_frequency
        print("Duration for {} events: ".format(n_events), duration_for_n_events)
        # number of samples for 10 revolutions
        n_samples_n_events = duration_for_n_events *sampling_frequency
        self.cut_signal_length = int(np.floor(n_samples_n_events/2)*2) # Ensure that the signals have an even length# Ensure that the signals have an even length# Ensure that the signals have an even length
        print("Cutting signals in length: ", self.cut_signal_length)

        self.cwr_meta_data = {"sampling_frequency":sampling_frequency}
        self.cwr_meta_data["expected_fault_frequencies"] = {test["mode"]: test["expected_fault_frequency"] for test_name,test in self.files_and_measurement_ids.items() if test["mode"] is not None}

        self.cwr_meta_data["expected_fault_frequencies"]["fr"] = rotation_rate # Also add the rotation rate and the FTF (Fundamental train frequency)
        self.cwr_meta_data["expected_fault_frequencies"]["ftf"] = 0.3983*rotation_rate

        self.db,self.client = make_db("cwr")
        self.db.drop_collection("raw")

    def create_document(self, time_series_data, fault_class, severity):
        doc = {"mode": fault_class,
               "severity": severity,
               "meta_data": self.cwr_meta_data,
               "time_series": list(time_series_data),
               }
        return doc

    def add_to_db(self,signal_segments,mode,severity):
        docs = [self.create_document(signal,mode,severity) for signal in signal_segments]

        # TODO: Add the test functionality here to make it around the healhty damage treshold
        self.db["raw"].insert_many(docs)

    def add_all_to_db(self):

        for file_name, info in self.files_and_measurement_ids.items():
            path_to_mat_file = cwr_path.joinpath(file_name)
            mat = loadmat(str(path_to_mat_file))
            signal = mat["X" + file_name[0:-4] + "_DE_time"].flatten()

            if file_name == "098.mat":
                signal = signal[::4].copy()  # Down sample the healthy data since it is sampled at a different sampling rate than the damaged data. 12kHz vs 48kHz

            percentage_overlap = 0.50
            signal_segments = overlap(signal, self.cut_signal_length, np.floor(self.cut_signal_length * percentage_overlap)) # Segments have half overlap

            print("Number of signal segments extracted: ", signal_segments.shape[0])

            self.add_to_db(signal_segments, info["mode"], info["severity"])

o = CWR()
o.add_all_to_db()
print("Signal length:" , o.cut_signal_length)



