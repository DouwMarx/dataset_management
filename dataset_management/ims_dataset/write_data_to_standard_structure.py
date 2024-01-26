import json
import pathlib
from datetime import datetime
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from dataset_management.ultils.write_data_in_standard_format import export_data_to_file_structure
from file_definitions import ims_path
from pypm.phenomenological_bearing_model.bearing_model import Bearing
from tqdm import tqdm
import plotly.graph_objects as go

class IMSTest(object):
    """
    Used to process the IMS data
    """

    def __init__(self, path_to_measurement_campaign: pathlib.Path, channel_info, rapid=False):
        self.folder_name = path_to_measurement_campaign.name  # The folder where the text files are stored

        self.channel_info = channel_info  # Info defined for the channel

        # Use only the channel names with verified faults such that "mode is not None"
        self.measurement_names = [channel_dict["measurement_name"] for channel_dict in self.channel_info]

        self.labeled_measurement_names = [channel_dict["measurement_name"] for channel_dict in self.channel_info if channel_dict["mode"] is not None]

        self.rotation_frequency = 2000 / 60  # rev/s  , Hz # From the IMS document

        self.rapid = rapid  # Reduce size for prototyping purposes

        self.measurement_paths = list(path_to_measurement_campaign.iterdir())
        if self.rapid:
            self.measurement_paths = np.random.choice(self.measurement_paths, size=1000, replace=False)

        # Make sure the record numbers follow the correct time stamps for different samples.
        self.time_stamps = sorted(
            [datetime.strptime(path.name, '%Y.%m.%d.%H.%M.%S') for path in self.measurement_paths])
        self.record_numbers = {self.time_stamps[i]: str(i + 1) for i in
                               range(len(self.time_stamps))}  # Notice 1-based indexing for record number

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
        measurement = pd.read_csv(file_path, sep="\t", names=self.measurement_names)
        if len(measurement.index) != self.n_samples_per_measurement:
            raise IndexError(
                "The number of samples in the file is different than expected: ".format(len(measurement.index)))
        return measurement

    def create_document_per_channel(self, filepath):
        dataframe_of_measurement_for_each_channel = self.read_file_as_df(filepath)

        list_of_docs = []
        for channel_id, channel_name in enumerate(dataframe_of_measurement_for_each_channel.columns):
            measurement_for_channel = dataframe_of_measurement_for_each_channel[channel_name].values
            doc = self.create_document(list(measurement_for_channel), channel_id, filepath)
            list_of_docs.append(doc)
        return list_of_docs

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
               "ims_channel_number": str(int(channel_id + 1)),  # IMS convention is 1-based channel numbering
               "measurement_name": info_for_channel["measurement_name"]
               }
        return doc

    def identify_ground_truth_health_state(self):
        "Plot the kurtosis as a function of time and cut the data into run-in, reference, uncertain and faulty based on the standard deviation for the kurtosis"
        # documents = Parallel(n_jobs=8)(delayed(self.create_document_per_channel)(path) for path in tqdm(self.measurement_paths))
        # # Flatten the list of lists
        # documents = [item for sublist in documents for item in sublist]
        #
        # df = pd.DataFrame(documents)
        # print(df)
        # # Compute the kurtosis for each sample row using the list in the "time_series" column
        # df["kurtosis"] = df["time_series"].apply(lambda x: pd.Series(x).kurtosis())
        # df["kurtosis"] = df["kurtosis"].astype(float)
        #
        # # Remove the time series data to save space
        # df = df.drop(columns=["time_series"])
        #
        # # # Temporarily write to pickle file to save compute
        # df.to_pickle("ims_test_{}.pkl".format(self.folder_name[0]))
        df = pd.read_pickle("ims_test_{}.pkl".format(self.folder_name[0]))

        split_dict = {}

        # Plot the kurtosis as a function of time
        for measurement in self.labeled_measurement_names:
            fig = go.Figure()
            df_for_channel = df[df["measurement_name"] == measurement]
            # Sort by time stamp
            df_for_channel = df_for_channel.sort_values(by="time_stamp")
            df_for_channel = df_for_channel.reset_index(drop=True)

            measurement_kurtosis= df_for_channel["kurtosis"]
            measurement_time_steps = df_for_channel["time_stamp"]
            median_kurtosis =  measurement_kurtosis.median()
            iqr_kurtosis =  measurement_kurtosis.quantile(0.75) - measurement_kurtosis.quantile(0.25)

            moving_median_kurtosis = measurement_kurtosis.rolling(window=10).median()
            # Split the data into run-in, reference, uncertain and faulty based on the kurtosis median and IQR
            # Get the first time index where the moving median is smaller than the median + 1*iqr
            thresh_ref_start = 1
            reference_start_idx  = moving_median_kurtosis[moving_median_kurtosis < median_kurtosis + thresh_ref_start*iqr_kurtosis].index[0]

            thresh_fault_start = 2
            # Get a point after the reference start point where the kurtosis is larger than the median + 2*iqr
            faulty_start_idx = moving_median_kurtosis[moving_median_kurtosis.index > reference_start_idx][moving_median_kurtosis > median_kurtosis + thresh_fault_start*iqr_kurtosis].index[0]

            # Make uncertain start 90% of the way between reference and faulty
            uncertain_start_idx = int(reference_start_idx + 0.9*(faulty_start_idx - reference_start_idx))

            points_of_interest = [reference_start_idx, uncertain_start_idx, faulty_start_idx]
            split_dict[measurement] = points_of_interest

            # Raise error is any of the points of interest are not in the data
            print("points of interest: ", points_of_interest, "for signal of length: ", len(measurement_kurtosis))

            # Plot the kurtosis as a function of time
            fig.add_trace(go.Scatter(x = measurement_time_steps, y = measurement_kurtosis, name=measurement))

            # Plot the moving median kurtosis as a function of time
            fig.add_trace(go.Scatter(x = measurement_time_steps,
                                     y = moving_median_kurtosis,
                                     name=measurement + " moving median",
                                     line=dict(color="black")))

            # Plot the median and multiples of the IQR as horizontal lines
            fig.add_trace(go.Scatter(x = [measurement_time_steps.iloc[0], measurement_time_steps.iloc[-1]], y = [median_kurtosis, median_kurtosis], name=measurement + " median", line=dict(color="black")))
            for i in range(1,3):
                fig.add_trace(go.Scatter(x = [measurement_time_steps.iloc[0], measurement_time_steps.iloc[-1]], y = [median_kurtosis + i*iqr_kurtosis, median_kurtosis + i*iqr_kurtosis], name=measurement + " {}x iqr".format(i), line=dict(color="black", dash="dash")))

            # Plot a vertical line for each point of interest. Dont show the label
            for point in points_of_interest:
                point = measurement_time_steps.iloc[point]
                fig.add_trace(go.Scatter(x = [point, point], y = [measurement_kurtosis.min(), measurement_kurtosis.max()], line=dict(color="black", dash="dash"), showlegend=False))
            fig.update_layout(title="Kurtosis for test: " + self.folder_name[0])

            # Label the axis
            fig.update_xaxes(title_text="Time")
            fig.update_yaxes(title_text="Kurtosis")
            fig.show()

            # Save the figure as html
            fig.write_html("ims_test_{}_{}_split.html".format(self.folder_name[0], measurement))

        # Save the split dict to json
        with open("ims_test_{}_split.json".format(self.folder_name[0]), "w") as f:
            json.dump(split_dict, f,default=str)


    def write_to_file(self, target_location=None):

        if target_location is None:
            target_location = pathlib.Path("/home/douwm/projects/PhD/code/biased_anomaly_detection/data")

        documents = Parallel(n_jobs=8)(delayed(self.create_document_per_channel)(path) for path in tqdm(self.measurement_paths))
        # Flatten the list of lists
        documents = [item for sublist in documents for item in sublist]

        df = pd.DataFrame(documents)

        split_dict =json.load(open("ims_test_{}_split.json".format(self.folder_name[0]), "r")) # Load the dictionary prescribing the reference, uncertain and faulty split

        for measurement in self.labeled_measurement_names:
            channel_df= df[df["measurement_name"] == measurement]
            channel_df = channel_df.sort_values(by="time_stamp")
            channel_df = channel_df.reset_index(drop=True)

            # Get the split points based on the kurtosis levels
            reference_start_idx, uncertain_start_idx, faulty_start_idx = np.array(split_dict[measurement]).astype(int)

            healthy_df = channel_df.iloc[reference_start_idx:uncertain_start_idx]
            faulty_df = channel_df.iloc[faulty_start_idx:]

            # Extract the time signals
            healthy_time_signals = np.vstack(healthy_df["time_series"].values)
            faulty_time_signals = np.vstack(faulty_df["time_series"].values)

            # Stack together and plot kurtosis to verify that the split is correct
            # stacked_time_signals = np.vstack([healthy_time_signals, faulty_time_signals])
            # kurtosis = scipy.stats.kurtosis(stacked_time_signals, axis=1)
            # plt.plot(kurtosis)
            # plt.show()

            # Make sure there is a channel dimension such that (batch, channel, time)
            healthy_time_signals = np.expand_dims(healthy_time_signals, axis=1)

            mode_name = "".join([str(mode) for mode in channel_df["mode"].unique()])
            faulty_time_signals = {mode_name : np.expand_dims(faulty_time_signals, axis=1)}

            export_data_to_file_structure(
                dataset_name="ims" + "_test" + self.folder_name[0] + "_" + measurement,
                healthy_data=healthy_time_signals,
                faulty_data_dict=faulty_time_signals,
                export_path=target_location,
                metadata=self.ims_meta_data
            )

if __name__ == "__main__":

    from dataset_management.ims_dataset.experiment_meta_data import channel_info, test_folder_names

    for key, folder in test_folder_names.items():
        folder_channel_info = channel_info[key]
        path_to_folder = ims_path.joinpath(folder)
        test_obj = IMSTest(path_to_folder, folder_channel_info, rapid=False)
        test_obj.write_to_file()
        # test_obj.identify_ground_truth_health_state()
