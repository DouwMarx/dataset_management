import numpy as np
import pandas as pd

from database_definitions import make_db
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

def get_metadata_from_csv():
    meta_data_SmithRandal2014 = pd.read_csv("meta_data_tablea2_Smith-Randal-2014.csv",keep_default_na=False)
    meta_data_SmithRandal2014.replace(np.nan,"0",inplace=True)
    return meta_data_SmithRandal2014


class CWR(object):
    """
    Used to add the CWR data to mongodb
    """

    def __init__(self):
        rotation_rate = 1772 / 60  # Rev/s
        self.sampling_frequency= 12000

        # For Ball failure mode having the lowest expected fault frequency
        lowest_expected_fault_frequency = 2.357 * rotation_rate
        n_events = 15
        # time required for n_events for highest fault frequency
        duration_for_n_events = n_events / lowest_expected_fault_frequency
        print("Duration for {} events: ".format(n_events), duration_for_n_events)
        # number of samples for 10 revolutions
        n_samples_n_events = duration_for_n_events *self.sampling_frequency
        self.cut_signal_length = int(np.floor(n_samples_n_events/2)*2) # Ensure that the signals have an even length# Ensure that the signals have an even length# Ensure that the signals have an even length
        print("Cutting signals in length: ", self.cut_signal_length)

        self.smith_randal_meta_data = get_metadata_from_csv()

        # Create a db for each of the operating conditions
        self.dbs = {}
        for oc in range(4): # There are a total of 4 operating conditions
            db,client = make_db("cwr_oc" + str(oc))
            db.drop_collection("raw") # Drop the collection if it already exists
            self.dbs[oc] = db

    def get_expected_fault_frequency_for_mode(self,mode, rpm):
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
    # def create_document(self, time_series_data, meta_data):
        doc = metadata.copy()
        doc["signal_data"] = list(time_series_data)
        return doc

    def add_to_db(self,signal_segments, meta_data):

        operating_condition = meta_data["oc"]

        docs = [self.create_document(signal,meta_data) for signal in signal_segments]

        # TODO: Add the test functionality here to make it around the healthy damage threshold
        self.dbs[operating_condition]["raw"].insert_many(docs) # Insert the documents into the db with the right operating condition


    def get_meta_data(self,stem):
        file_name_number = int(stem)

        r = self.smith_randal_meta_data

        # Find the row and column position in the dataframe where the number occurs
        row, column = np.where(r == file_name_number)
        fault_width = r.iloc[row, 0].values[0]
        hp = r.iloc[row, 1].values[0]
        rpm = r.iloc[row, 2].values[0]
        mode = str(r.columns[column])
        expected_fault_frequency = self.get_expected_fault_frequency_for_mode(mode, rpm)

        print("rpm = ", rpm, " fault_width = ", fault_width, " expected_fault_frequency", expected_fault_frequency)

        meta_data = {   "severity":fault_width,
                        "oc":int(hp), # Operating condition
                        "rpm":int(rpm),
                        "expected_fault_frequency":float(expected_fault_frequency) if expected_fault_frequency != None else None,
                        "mode":mode,
                        "dataset_number":file_name_number,
                        "sampling_frequency": self.sampling_frequency# Hz
        }
        return meta_data


    def add_all_to_db(self):
        # loop trough all files in the pathlib path directory
        for file_name in cwr_path.glob("*.mat"):
            meta_data = self.get_meta_data(file_name.stem)
            path_to_mat_file = cwr_path.joinpath(file_name.name)
            mat = loadmat(str(path_to_mat_file)) # Load the .mat file
            key = [key for key in mat.keys() if "DE" in key][0]# Here we select the drive-end measurements
            signal = mat[key].flatten() # Here we select the drive-end measurements

            if meta_data["severity"] == 0: # The healthy data is sampled at a higher rate
                signal = signal[::4].copy()  # Down sample the healthy data since it is sampled at a different sampling rate than the damaged data. 12kHz vs 48kHz
                print("Down sampling healthy data, dataset ", meta_data["dataset_number"])

            percentage_overlap = 0.50
            signal_segments = overlap(signal, self.cut_signal_length, np.floor(self.cut_signal_length * percentage_overlap)) # Segments have half overlap

            print("Number of signal segments extracted: ", signal_segments.shape[0])

            self.add_to_db(signal_segments, meta_data)

o = CWR()
o.add_all_to_db()
print("Signal length:" , o.cut_signal_length)




