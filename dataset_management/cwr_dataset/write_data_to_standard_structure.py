import pathlib
from itertools import product

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, decimate
from scipy.io import loadmat
from dataset_management.ultils.time_frequency import get_required_signal_length_for_required_number_of_events
from dataset_management.ultils.write_data_in_standard_format import export_data_to_file_structure
from file_definitions import cwr_path


def get_cwru_meta_data_from_csv():
    # Load the data table from the Smith and Randal paper with dataset information
    # This table is also directly available from the source of the dataset

    # All data should be strings
    meta_data = pd.read_csv("meta_data_table.csv", dtype=str)

    value_cols = ['Inner Race Fault', 'Ball Fault', 'Outer Race Fault: Centre', 'Outer Race Fault: Orthogonal',
                  'Outer Race: Opposite ', 'Reference']
    id_cols = [col for col in meta_data.columns if col not in value_cols]
    # Convert the fault mode columns to a single column
    meta_data = pd.melt(meta_data, id_vars=id_cols, value_vars=value_cols, var_name="Fault Mode", value_name="File Number")

    # Return only the rows where the file number is not empty, reset the index
    meta_data = meta_data[meta_data["File Number"].notnull()]
    meta_data = meta_data.reset_index(drop=True)
    return meta_data


class CWR(object):
    """
    Used read and write the CWR dataset
    """

    def __init__(self, percentage_overlap=0, required_average_number_of_events_per_rev=30, name="CWR"):

        # Parameters of the dataset
        self.percentage_overlap = percentage_overlap

        if self.percentage_overlap > 0:
            print("Warning: using overlap can lead to data leakage")

        self.name = name
        self.sampling_frequency = 12000

        # Load the meta data as extracted from Smith and Randal table A2.
        self.sample_labels = get_cwru_meta_data_from_csv()

        # Retrieve the unique rpms from the meta data "Shaft speed [rpm]"
        rpms = np.unique(self.sample_labels["Shaft speed [rpm]"].values.astype(int))
        rpms = np.sort(rpms)  # Sort the rpms in ascending order
        mean_rpm = np.mean(rpms)
        minimum_number_of_events_occuring_per_revolution = 2.357  # The slowest fault frequency is 2.357 Hz, see Smith and Randal Table 2

        # Compute the required signal length to ensure that there are at least the required number of fault events per sample
        self.cut_signal_length = get_required_signal_length_for_required_number_of_events(mean_rpm,
                                                                                          minimum_number_of_events_occuring_per_revolution,
                                                                                          self.sampling_frequency,
                                                                                          required_average_number_of_events_per_rev
                                                                                          )
        print("Signals cut to length: {}, ensuring that there are at least {} events per revolution".format(
            self.cut_signal_length, required_average_number_of_events_per_rev))

        self.n_faults_per_revolution = {mode: self.get_expected_fault_frequency_for_mode(mode, 60) for mode in
                                        ["inner", "outer", "ball"]}  # 60 RPM = 1 Hz = 1 rev/s

        self.dataset_meta_data = {"signal_length": self.cut_signal_length,
                                  "sampling_frequency": self.sampling_frequency,
                                  "n_faults_per_revolution": self.n_faults_per_revolution,
                                  "long_name": "Case Western  Reserve  University  Bearing",
                                  "cut_signal_length": self.cut_signal_length,
                                  "channel_names": ["DE", "FE", "BA"],
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
        Pull the meta data for a particular file from the CSV file with the meta data
        """
        # file_name_number = int(file_stem)
        #
        # r = self.meta_data_df
        #
        # # Find the row and column position in the dataframe where the number of the meaurement occurs
        # row, column = np.where(r == file_name_number)
        # fault_width = r.iloc[row, 0].values[0]  # Fault width is in the first column
        # hp = r.iloc[row, 1].values[0]  # Power (Operating condition) is in the second column
        # rpm = r.iloc[row, 2].values[0]  # Speed is in the third column
        # mode = str(r.columns[column].values[0])  # The mode is in the column name
        #
        # sample_meta_data = {"severity": fault_width,
        #                     "oc": int(hp),  # Operating condition
        #                     "snr": "inf", #  No noise is added
        #                     "mode": mode,
        #                     "rpm": int(rpm),
        #                     "dataset_number": file_name_number,
        #              }

        sample_meta_data = self.sample_labels[self.sample_labels["File Number"] == file_stem].iloc[0].to_dict()

        # rpm = "Shaft speed [rpm]"
        # mode
        # expected_fault_frequency = self.get_expected_fault_frequency_for_mode(sample_meta_data["mode"], rpm)
        derived_meta_data = {
            # "expected_fault_frequency": float(
            #     expected_fault_frequency) if expected_fault_frequency != None else None,
            # "all_expected_fault_frequencies": {mode: self.get_expected_fault_frequency_for_mode(mode, sample_meta_data["rpm"]) for
            #                                    mode in ["inner", "outer", "ball"]},
        }

        return sample_meta_data, derived_meta_data

    def down_sample_reference(self, signal):
        # The reference was sampled only at 48KHz, need to downsample to 12KHz

        downsampling_factor = 4
        # Calculate the cutoff frequency based on the downsampling factor
        nyquist = 0.5 * 1  # Nyquist frequency before downsampling
        cutoff = 0.5 / downsampling_factor  # Cutoff frequency after downsampling

        # Apply the low-pass filter to the signal to avoid aliasing
        b, a = butter(6, cutoff / nyquist, btype='low')
        filtered_signal = lfilter(b, a, signal)
        # Downsample the filtered signal by the factor of 4
        signal = decimate(filtered_signal, downsampling_factor)
        return signal

    def segment_signals(self, signal):
        # Change the dimension of the data to now be (batch_size =1, n_channels, signal_length)
        data = np.expand_dims(signal, axis=(0, 1))  # (batch_size =1, n_channels = 1, signal_length)
        # Reshape data by cutting the signal into segments of length self.cut_signal_length and thereby adding to the batch dimension
        n_segments = data.shape[2] // self.cut_signal_length
        data = data[:, :, :n_segments * self.cut_signal_length]
        data = data.reshape(n_segments, 1, self.cut_signal_length)
        return data

    def load_mat_file(self, file_number):
        sample_meta_data, derived_meta_data = self.get_label(file_number)
        path_to_mat_file = cwr_path.joinpath(str(file_number) + ".mat")
        mat_file = loadmat(str(path_to_mat_file))  # Load the .mat file

        print("")
        datasets = []
        present_channels = [key for key in mat_file.keys() if
                            any(channel in key for channel in self.dataset_meta_data["channel_names"])]
        present_channel_names = [channel for channel in self.dataset_meta_data["channel_names"] if
                                 any(channel in key for key in mat_file.keys())]
        for channel, channel_name in zip(present_channels, present_channel_names):
            # Notify when the number in the channel name after X and before _ does not match sample_meta_data["dataset_number"] # This is the case for the reference data?
            if channel.split("X")[1].split("_")[0] != sample_meta_data["File Number"]:
                print("Warning: channel name does not match dataset number for file: " + str(
                    file_number) + " channel: " + str(channel) + " file_number: " + str(sample_meta_data))

            signal = mat_file[channel].flatten()  # Select measurements at a given location
            data = self.segment_signals(signal)
            channel_specific_meta_data = sample_meta_data.copy()
            channel_specific_meta_data.update({"Measurement Location": channel_name})
            datasets.append((data, channel_specific_meta_data, derived_meta_data))

            # No reference was sampled only at is available at 12KHz, so if Refence add another set by downsampling repeating steps above
            if sample_meta_data["Fault Mode"] == "Reference":
                signal_ds = self.down_sample_reference(signal)
                data = self.segment_signals(signal_ds)
                channel_specific_meta_data = sample_meta_data.copy()
                channel_specific_meta_data.update({"Measurement Location": channel_name})
                # Modify the Sampling Rate [kHz] to 12
                channel_specific_meta_data.update({"Sampling Rate [kHz]": "12"})
                datasets.append((data, channel_specific_meta_data, derived_meta_data))

        return datasets

    def load_all_samples(self):
        # loop through all files in the pathlib path directory
        labels = []
        signals = []

        for file_name in self.sample_labels["File Number"].values:
            datasets = self.load_mat_file(file_name)
            labels += [dataset[1] for dataset in datasets]
            signals += [dataset[0] for dataset in datasets]

        labels = pd.DataFrame(labels)

        return signals, labels


def get_data_entry(labels, signals, operating_condition, severity, mode, channel):
    # Return the index of the data entry with the given operating condition and severity if no data is found then return None
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


min_average_events_per_rev = 8
# min_average_events_per_rev = 20
cwr_data = CWR(percentage_overlap=0, required_average_number_of_events_per_rev=min_average_events_per_rev,
               name="cwr" + str(min_average_events_per_rev))

# Load the signals and write them to the standard file structure
signals, labels = cwr_data.load_all_samples()

all_reference_datasets = labels[labels["Fault Location"] =="NONE"]
all_faulty_datasets = labels[labels["Fault Location"] !="NONE"]

operating_condition_parameters = ["Motor load [hp]", "Sampling Rate [kHz]", "Measurement Location"]
fault_condition_parameters = ["Fault Location", "Fault Width [mm]", "Fault Mode"]
for index, healthy_data_row in all_reference_datasets.iterrows():
    print(healthy_data_row.dtype)
    healthy_data = signals[index]
    # Dataset name includes "Motor load [hp]", "Sampling Rate [kHz]" and "Measurement Location"
    dataset_name = "CWRU_{}_".format(min_average_events_per_rev) + "_".join(key + "_" + healthy_data_row[key] for key in operating_condition_parameters)


    # Get the corresponding faulty data that has the same operating condition and measurement location as the healthy data
    # "Sampling Rate [kHz]", "Motor load [hp]", "Measurement Location"
    corresponding_faulty_data = all_faulty_datasets[
        (all_faulty_datasets["Sampling Rate [kHz]"] == healthy_data_row["Sampling Rate [kHz]"]) & (
                    all_faulty_datasets["Motor load [hp]"] == healthy_data_row["Motor load [hp]"]) & (
                    all_faulty_datasets["Measurement Location"] == healthy_data_row["Measurement Location"])]

    faulty_data_dict = {}
    for index, faulty_data_row in corresponding_faulty_data.iterrows():
        faulty_data = signals[index]
        # Faulty names
        faulty_name = "_".join(key + "_" + faulty_data_row[key] for key in fault_condition_parameters)
        faulty_data_dict[faulty_name] = faulty_data

    export_data_to_file_structure(dataset_name=dataset_name,
                                  healthy_data=healthy_data,
                                  faulty_data_dict=faulty_data_dict,
                                  export_path=pathlib.Path(
                                      "/home/douwm/projects/PhD/code/biased_anomaly_detection/data"),
                                  metadata=cwr_data.dataset_meta_data
                                  )

labels = get_cwru_meta_data_from_csv()
# Save a copy of the processed labels
labels.to_csv("CWRU_processed_labels.csv")