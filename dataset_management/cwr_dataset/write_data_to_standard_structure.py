import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy.signal import decimate
from scipy.io import loadmat
from dataset_management.ultils.time_frequency import get_required_signal_length_for_required_number_of_events, \
    get_number_of_fault_events_for_segment_length
from dataset_management.ultils.write_data_in_standard_format import export_data_to_file_structure
from file_definitions import cwr_path, biased_anomaly_detection_path

"""
The case western Reserve dataset has multiple sampling rates, operating conditions, fault modes, fault severities and measurement locations.
This script tries to organise the data.
Note that "healthy data" at 12KHz is obtained by down sampling the 48KHz reference data, since no 12KHz reference data is available.
"""

def get_cwru_meta_data_from_csv():
    # Load the data table from the Smith and Randal paper with dataset information
    # This table is also directly available from the website source of the dataset
    this_file = pathlib.Path(__file__)

    meta_data = pd.read_csv(this_file.parent.joinpath("meta_data_table.csv"), dtype=str)  # All data is stored as strings
    value_cols = ['Inner Race Fault',
                  'Ball Fault',
                  'Outer Race Fault: Centre',
                  'Outer Race Fault: Orthogonal',
                  'Outer Race: Opposite ',
                  'Reference']
    id_cols = [col for col in meta_data.columns if col not in value_cols] # The columns that are not fault modes
    # Convert the fault mode columns to a single column
    meta_data = pd.melt(meta_data, id_vars=id_cols, value_vars=value_cols, var_name="Fault Mode",
                        value_name="File Number")
    # Return only the rows where the file number is not empty (i.e. a measurement was taken)
    meta_data = meta_data[meta_data["File Number"].notnull()]
    meta_data = meta_data.reset_index(drop=True)
    return meta_data # The re-shaped meta data table


class CWR(object):
    """
    Used to read and write the CWR dataset to a standard file structure
    """

    def __init__(self, required_average_number_of_events_per_segment=30, name="CWR", data_path=cwr_path,overlap=0.0):

        """
        :param required_average_number_of_events_per_segment:   The average number of fault events required per signal segment. If a fault event is not present in a segment, a model can not be expected to detect it.
        :param name:  The name of the dataset
        """

        self.name = name
        self.data_path = data_path
        self.overlap = overlap

        if overlap>0:
            #  raise a warning that splitting should be done responsibly when overlap is used
            warnings.warn("Do not use randomized splitting when overlap is used. This will cause data leakage. Instead, use a fixed split and discard the overlap.")

        self.sample_labels = get_cwru_meta_data_from_csv() # Load the meta data as extracted from Smith and Randal table A2 and present on website

        # Find the mean RPM so that we can compute the required signal length to ensure that there are at least a minimum number of required number of fault events per sample
        rpms = np.unique(self.sample_labels["Shaft speed [rpm]"].values.astype(int))  # Unique RPMs
        rpms = np.sort(rpms)  # Sort the rpms in ascending order for nicer tables
        mean_rpm = np.mean(rpms)
        minimum_number_of_events_occurring_per_revolution = 2.357  # The slowest expected fault frequency is 2.357 Hz, see Smith and Randal Table 2

        # Compute the required signal length to ensure that there are at least the required number of fault events per sample
        self.cut_signal_length_for_12k = get_required_signal_length_for_required_number_of_events(mean_rpm,
                                                                                                  minimum_number_of_events_occurring_per_revolution,
                                                                                                  12000,  # TODO: The sampling rate is hard-coded and does not apply to the datasets sampled at a higher rate.
                                                                                                  required_average_number_of_events_per_segment
                                                                                                  )
        print("Signals cut to length: {}, ensuring that there are at least {} events per revolution".format(
            self.cut_signal_length_for_12k, required_average_number_of_events_per_segment))
        print("")

        worst_case_faults_per_segment = get_number_of_fault_events_for_segment_length(min(rpms), minimum_number_of_events_occurring_per_revolution, 12000, self.cut_signal_length_for_12k)
        best_case_faults_per_segment = get_number_of_fault_events_for_segment_length(max(rpms), minimum_number_of_events_occurring_per_revolution, 12000, self.cut_signal_length_for_12k)
        print("Worst case  and best case fault events per segment: ", worst_case_faults_per_segment, best_case_faults_per_segment)

        self.n_faults_per_revolution = {mode: self.get_expected_fault_frequency_for_mode(mode, 60) for mode in
                                        ["inner", "outer", "ball"]}  # 60 RPM = 1 Hz = 1 rev/s

        self.dataset_meta_data = {
            "n_faults_per_revolution": self.n_faults_per_revolution,  # Should be sampling rate independent
            "long_name": "Case Western  Reserve  University  Bearing",
            "cut_signal_length_for_12k": self.cut_signal_length_for_12k,  # Remember that this length is sampling rate dependent
            "signal_length_for_12k": self.cut_signal_length_for_12k,  # Remove duplicate information
            "channel_names": ["DE", "FE", "BA"],  # Measurements are taken at the drive end, fan end and base
            "_id": "meta_data"
        }

    def get_expected_fault_frequency_for_mode(self, mode, rpm):
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

    def get_label(self, file_stem):
        """
        Pull the meta data and other associated "target" information for a particular file from the CSV file with the meta data
        """

        # Read a row from the CSV file with the meta data
        sample_meta_data = self.sample_labels[self.sample_labels["File Number"] == file_stem].iloc[0].to_dict()

        shaft_speed = int(sample_meta_data["Shaft speed [rpm]"])
        # Retrieve the expected fault frequency
        if "ball" in sample_meta_data["Fault Mode"].lower():
            expected_fault_frequency = self.get_expected_fault_frequency_for_mode("ball",  shaft_speed)
        elif "inner" in sample_meta_data["Fault Mode"].lower():
            expected_fault_frequency = self.get_expected_fault_frequency_for_mode("inner",  shaft_speed)
        elif "outer" in sample_meta_data["Fault Mode"].lower():
            expected_fault_frequency = self.get_expected_fault_frequency_for_mode("outer",  shaft_speed)
        else:
            expected_fault_frequency = None

        # Update the meta data with the expected fault frequency
        sample_meta_data.update({"Fault Frequency": expected_fault_frequency})

        derived_meta_data = {
        }  # TODO: Add derived meta data like expected/approximate fault frequencies etc.

        return sample_meta_data, derived_meta_data

    def down_sample_reference(self, signal):
        # The reference condition was sampled only at 48KHz (not 12KHz), and is be downsampled to be able to compare with 12KHz faulty data
        downsampling_factor = 4
        # Downsample the filtered signal by the factor of 4, ensuring that an anti-aliasing filter is applied
        signal = decimate(signal, downsampling_factor)
        return signal

    def segment_signals(self, signal, fs_is_48KHz=False):

        if fs_is_48KHz:
            cut_length = self.cut_signal_length_for_12k * 4 # Cut signal length was computed for 12KHz data, so we need to multiply by 4 to get the correct length for 48KHz data
        else:
            cut_length = self.cut_signal_length_for_12k

        step_size = int(cut_length * (1 - self.overlap))  # step size derived from overlap
        n_segments = (len(signal) - cut_length) // step_size + 1  # number of segments considering overlap

        segments = np.zeros((n_segments, 1, cut_length))  # initialize array to store segments (Notice channel dimension is 1 for torch)

        for i in range(n_segments):
            start = i * step_size
            end = start + cut_length
            segments[i, 0, :] = signal[start:end]  # extract segment and store in array

        return segments

    def get_data_per_channel_from_mat_file(self, file_number):
        sample_meta_data, derived_meta_data = self.get_label(file_number)  # First load the meta data from the CSV file

        path_to_mat_file = self.data_path.joinpath(str(file_number) + ".mat")
        mat_file = loadmat(str(path_to_mat_file))  # Load the .mat file

        sampling_rate_associated_with_file = sample_meta_data["Sampling Rate [kHz]"]

        # Find the channels that are present in the mat file and the corresponding channel names
        present_measurement_locations = [key for key in mat_file.keys() if
                            any(channel in key for channel in self.dataset_meta_data["channel_names"])] # Channel names as in the mat file
        present_measurement_location_names = [channel for channel in self.dataset_meta_data["channel_names"] if
                                 any(channel in key for key in mat_file.keys())] # Simpler channel names

        datasets = []  # Each file might contrain multiple measurement channels (measurement locations), so we loop through all channels and add each as a "dataset"
        for measurement_location, measurement_location_name in zip(present_measurement_locations, present_measurement_location_names):
            # Notify when the number in the channel name after X and before _ does not match sample_meta_data["dataset_number"]
            # This inconsistency is present for the reference data
            if measurement_location.split("X")[1].split("_")[0] != sample_meta_data["File Number"]:
                print("Note: channel name in mat file does not match dataset number for file: " + str(
                    file_number) + " channel: " + str(measurement_location) + " file_number: " + str(sample_meta_data))

            # Select measurements at a given measurement location/channel
            signal = mat_file[measurement_location].flatten()

            if sampling_rate_associated_with_file == "48":
                data = self.segment_signals(signal, fs_is_48KHz=True)  # Cut into segments
            else:
                data = self.segment_signals(signal, fs_is_48KHz=False)

            channel_specific_meta_data = sample_meta_data.copy()
            channel_specific_meta_data.update({"Measurement Location": measurement_location_name})
            datasets.append((data, channel_specific_meta_data, derived_meta_data))

            # No reference is available for 12KHz data, so if the data is from the reference condition, the 48KHz reference data is downsampled and added to the list of datasets
            # This means that a given 48KHz reference sample might be used multiple times, once at 48KHz and once at 12KHz
            if sample_meta_data["Fault Mode"] == "Reference":
                signal_ds = self.down_sample_reference(signal)
                channel_specific_meta_data_ref = sample_meta_data.copy()
                channel_specific_meta_data_ref.update({"Sampling Rate [kHz]": "12"})  # Update the sampling rate to 12 since it was downsampled

                data_ds = self.segment_signals(signal_ds, fs_is_48KHz=False)
                channel_specific_meta_data_ref.update({"Measurement Location": measurement_location_name})

                datasets.append((data_ds, channel_specific_meta_data_ref, derived_meta_data))
        return datasets

    def load_all_mat_files(self):
        labels = []
        signals = []

        for file_name in self.sample_labels["File Number"].values:  # Loop through all files measured files
            datasets = self.get_data_per_channel_from_mat_file(file_name)
            signals += [dataset[0] for dataset in datasets]
            labels += [dataset[1] for dataset in
                   datasets]  # Datasets were saved in tuples of (data, channel_meta_data, derived_meta_data)
        df = pd.DataFrame(labels)
        # Add the signals to the dataframe
        df["Signals"] = signals
        return df

def get_data_entry(labels, signals, operating_condition, severity, mode, channel):
    # Return the index of the data entry with the given operating condition and severity if no data is found then return None
    # labels is a pandas dataframe with the meta data for each data entry
    # signals is a list of numpy arrays with the signals
    data_entry = labels[
        (labels["oc"] == operating_condition) & (labels["severity"] == severity) & (labels["mode"] == mode) & (
                labels["channel"] == channel)]
    if len(data_entry) == 0:
        print("No data found for operating condition: " + str(operating_condition) + " and severity: " + str(
            severity) + " and mode: " + str(mode) + " and channel: " + str(channel))
        return None
    elif len(data_entry) > 1:
        print(
            "Multiple data entries found for operating condition: oc" + str(operating_condition) + " severity: " + str(
                severity) + " mode: " + str(mode))
        return None
    else:
        return signals[data_entry.index[0]]

def write_cwru_to_standard_file_structure(min_average_events_per_rev):
    # Writes the data to the standard file structure
    cwr_data = CWR(
        required_average_number_of_events_per_segment=min_average_events_per_rev,
        name="cwr" + str(min_average_events_per_rev),
       overlap=0.0 # Notice no overlap is used in this variant
    )

    # Load all signals into memory
    df = cwr_data.load_all_mat_files()

    all_reference_datasets = df[df["Fault Location"] == "NONE"]
    all_faulty_datasets = df[df["Fault Location"] != "NONE"]  # notice the negation

    operating_condition_parameters = ["Motor load [hp]", "Sampling Rate [kHz]", "Measurement Location"]
    fault_condition_parameters = ["Fault Location", "Fault Width [mm]", "Fault Mode"]
    for index, healthy_data_row in all_reference_datasets.iterrows():  # Loop through all the reference data and find the corresponding faulty data for different fault conditions
        healthy_data = df["Signals"][index]
        # Dataset name should include things like "Motor load [hp]", "Sampling Rate [kHz]" and "Measurement Location"
        dataset_name = "CWRU_{}_".format(min_average_events_per_rev) + "_".join(
            key + "_" + healthy_data_row[key] for key in operating_condition_parameters)

        # Get the corresponding faulty data that has the same operating condition and measurement location as the healthy data
        corresponding_faulty_data = all_faulty_datasets[
            (
                    all_faulty_datasets["Sampling Rate [kHz]"] == healthy_data_row["Sampling Rate [kHz]"]) & (
                    all_faulty_datasets["Motor load [hp]"] == healthy_data_row["Motor load [hp]"]) & (
                    all_faulty_datasets["Measurement Location"] == healthy_data_row["Measurement Location"])
            ]

        faulty_data_dict = {}  # Build a dictionary mapping each faulty data mode to its data
        for index, faulty_data_row in corresponding_faulty_data.iterrows():
            faulty_data = df["Signals"][index]
            # Faulty names
            faulty_name = "_".join(key + "_" + faulty_data_row[key] for key in fault_condition_parameters)
            faulty_data_dict[faulty_name] = faulty_data

        sampling_frequency = int(healthy_data_row["Sampling Rate [kHz]"]) * 1000

        meta_data = cwr_data.dataset_meta_data.copy()
        meta_data.update({"sampling_frequency": sampling_frequency})

        export_data_to_file_structure(dataset_name=dataset_name,
                                      healthy_data=healthy_data,
                                      faulty_data_dict=faulty_data_dict,
                                      export_path=biased_anomaly_detection_path,
                                      metadata=meta_data
                                      )


def get_cwru_data_frame(min_average_events_per_rev, overlap, path_to_write=None, data_path=cwr_path):
    # Writes the data to the standard file structure
    cwr_data = CWR(
        required_average_number_of_events_per_segment=min_average_events_per_rev,
        name="cwr" + str(min_average_events_per_rev),
        data_path=data_path,
        overlap=overlap
    )

    # Load all signals into memory
    df = cwr_data.load_all_mat_files()

    # Write to pickle
    if path_to_write is not None:
        df.to_pickle(path_to_write)
    return df


if __name__ == "__main__":
    min_average_events_per_rev = 8
    write_cwru_to_standard_file_structure(min_average_events_per_rev) # If you want folders with each operating condition

    # raw_directory = pathlib.Path(__file__).parent.joinpath("raw_data")
    # write_directory = pathlib.Path(__file__).parent.joinpath("processed_data")
    # # Make write directory if it does not exist
    # if not write_directory.exists():
    #     write_directory.mkdir()
    #
    # get_cwru_data_frame(min_average_events_per_rev,
    #                     overlap=0.5,
    #                     data_path= raw_directory,
    #                     # data_path= cwr_path,
    #                     path_to_write=write_directory.joinpath("cwr_dataframe.pkl")) # One big dataframe with everything
