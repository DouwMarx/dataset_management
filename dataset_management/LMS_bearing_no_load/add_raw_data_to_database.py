import numpy as np
from database_definitions import make_db
from dataset_management.ultils import processing, create_noisy_copies_of_dataset, compute_features
from dataset_management.ultils.pul_data_from_db import get_data_from_db_and_save_to_file
from dataset_management.ultils.time_frequency import get_required_signal_length_for_required_number_of_events
from tqdm import tqdm
from scipy.io import loadmat
from file_definitions import lms_path
import matplotlib.pyplot as plt


class LMS(object):
    """
    Used to add the IMS data to mongodb
    """

    def __init__(self, percentage_overlap=0.5):
        self.datasets = {
            # "SOR2.mat": {"mode": "outer",
            #              "trail": 2,
            #              "severity": 1,
            #              "oc_ranges_mega": [
            #                  [0.847, 1.214],
            #                  [1.658, 1.967],
            #                  [2.290, 2.720],
            #                  [3.078, 3.417],
            #              ]
            #              },
            "SOR1.mat": {"mode": "outer",
                         "trail": 1,
                         "severity": 1,
                         "oc_ranges_mega": [
                             [0.637, 0.985],
                             [1.402, 1.709],
                             [2.147, 2.464],
                             [2.793, 3.112],
                         ]
                         },
            # "MIR3.mat": {"mode": "inner",
            #              "trail": 3,
            #              "severity": 2,
            #              "oc_ranges_mega": [
            #                  [1.533,1.830],
            #                  [2.322,2.651],
            #                  [3.251,3.607],
            #                  [3.939,4.244],
            #              ]
            #              },
            # "MIR2.mat": {"mode": "inner",
            #              "trail": 2,
            #              "severity": 2,
            #              "oc_ranges_mega": [
            #                  [1.121,1.512],
            #                  [1.981,2.360],
            #                  [2.897,3.276],
            #                  [3.913,4.254],
            #              ]
            #              },
            "MIR1.mat": {"mode": "inner",
                         "trail": 1,
                         "severity": 2,
                         "oc_ranges_mega": [
                             [0.885, 1.216],
                             [1.524,2.011],
                             [2.498,2.870],
                             [3.352,3.729],
                         ]
                         },
            # "LOR3.mat": {"mode": "outer",
            #              "trail": 3,
            #              "severity": 2,
            #              "oc_ranges_mega": [
           #                  [0.710, 1.110],
            #                  [1.614,1.849],
            #                  [2.297,2.621],
            #                  [3.074,3.382],
            #              ]
            #              },
            # "LOR2.mat": {"mode": "outer",
            #              "trail": 2,
            #              "severity": 2,
            #              "oc_ranges_mega": [
            #                  [0.782, 1.152],
            #                  [1.661,2.146],
            #                  [2.582,2.928],
            #                  [3.645,4.039],
            #              ]
            #              },
            "LOR1.mat": {"mode": "outer",
                         "trail": 1,
                         "severity": 2,
                         "oc_ranges_mega": [
                             [0.883, 1.230],
                             [1.698, 2.076],
                             [2.551, 2.775],
                             [3.386, 3.678],
                         ]
                         },
            # "HEA2.mat": {"mode": "healthy",
            #              "trail": 2,
            #              "severity": 0,
            #              "oc_ranges_mega": [
            #                  [0.800, 1.178],
            #                  [1.531, 1.853],
            #                  [2.380, 2.732],
            #                  [3.132, 3.482],
            #              ]
            #              },
            "HEA1.mat": {"mode": "healthy",
                         "trail": 1,
                         "severity": 0,
                         "oc_ranges_mega": [
                             [0.890, 1.278],
                             [1.587, 1.931],
                             [2.463, 2.806],
                             [3.173, 3.548],
                         ]
                         },
        }

        self.percentage_overlap = percentage_overlap

        self.sampling_frequency = 51200  # Sampling rate derived from increment parameter in the "x_values" field

        # Get the length for the slowest speed, the slowest fault frequency to have 10 fault events
        # min_rpm = 610
        mean_rpm = 1800
        min_events_per_rev = 2.196  # Ball fault frequency
        min_events = 8
        self.cut_signal_length = get_required_signal_length_for_required_number_of_events(mean_rpm, min_events_per_rev,
                                                                                          self.sampling_frequency, min_events)
        print("Cut signal length: {}".format(self.cut_signal_length))

        self.db, self.client = make_db("lms")

        # Drop the previous data
        self.db.drop_collection("raw")
        self.db.drop_collection("meta_data")

        # From SKF website for bearing 2206 EKTN9
        self.n_events_per_rev = {
            # "cage": 0.394,
            "ball": 2.196,
            "inner": 7.276,
            "outer": 4.724,
            "healthy": None
        }

        # Add a document to the db with _id meta_data to store the meta data
        self.db["meta_data"].insert_one({"_id": "meta_data",
                                         "sampling_frequency": self.sampling_frequency,
                                         "n_faults_per_revolution": self.n_events_per_rev,
                                         "dataset_name": "CWR",
                                         "percentage_overlap": self.percentage_overlap,
                                         "cut_signal_length": self.cut_signal_length,
                                         })

    def get_accelerometer_signal(self,dataset_dict_key):
        path_to_mat_file = lms_path.joinpath(dataset_dict_key)
        segments_to_extract = self.datasets[dataset_dict_key]["oc_ranges_mega"]

        mat = loadmat(str(path_to_mat_file), matlab_compatible=True, simplify_cells=True)
        accelerometer_signals = mat["Signal_3"]["y_values"]["values"]  # Signal_3 is the accelerometer
        speed_signal = mat["Signal_1"]["y_values"]["values"]  # Signal_1 is the speed signal
        speed_correction_factor = mat["Signal_1"]["y_values"]["quantity"]["unit_transformation"]["factor"]

        n_events_per_rev = self.n_events_per_rev[self.datasets[dataset_dict_key]["mode"]]

        signals_at_oc = []
        for oc, samples in enumerate(segments_to_extract[1:]): # Do not use the first segment at the lowest speed
            start_id = int(samples[0] * 1e6)
            end_id = int(samples[1] * 1e6)

            speed_samples = speed_signal[start_id:end_id] * speed_correction_factor  # In RPM
            median_speed = np.median(speed_samples)
            speed_fluctuation = np.std(speed_samples)
            print("Median speed: ", median_speed, "RPM", "speed fluctuation: ", speed_fluctuation, "RPM")

            for accelerometer in [1]: # TODO: Notice we are only using the first accelerometer
            # for accelerometer in [0, 1]:
                sig = accelerometer_signals[:, accelerometer]  # Numbering is not necessarily correct

                signals_at_oc.append({"signal": sig[start_id:end_id],  # Use only the cut out segment
                                "rpm": median_speed,
                                "oc":oc,
                                "mode": self.datasets[dataset_dict_key]["mode"],
                                "severity": self.datasets[dataset_dict_key]["severity"],
                                "trail": self.datasets[dataset_dict_key]["trail"],
                                "rpm_fluctuation": speed_fluctuation,
                                "accelerometer_number": accelerometer,
                                "expected_fault_frequency": n_events_per_rev * median_speed / 60 if n_events_per_rev is not None else None,
                                "all_expected_fault_frequencies": {k: v * median_speed / 60 for k, v in  self.n_events_per_rev.items() if v is not None},
                                "snr": 0,
                                })
        return signals_at_oc


    def add_to_db(self, dataset_dict_key):

        signals_at_oc = self.get_accelerometer_signal(dataset_dict_key)

        documents_for_dataset = []
        for signal_at_oc in signals_at_oc:
            signal_segments = processing.overlap(signal_at_oc["signal"], self.cut_signal_length, self.percentage_overlap)
            template = signal_at_oc.copy()
            template["signal"] = []

            # Create a document for each signal segment, duplicating the meta data
            for signal_segment in signal_segments:
                doc = template.copy()
                doc["time_series"] = list(signal_segment)
                documents_for_dataset.append(doc)

        # TODO: Add the test functionality here to make it around the healthy damage treshold
        self.db["raw"].insert_many(documents_for_dataset)

    def add_all_to_db(self):
        for key, val in tqdm(self.datasets.items()):  # TODO: Notice use of last 10 seconds of data
            self.add_to_db(key)


    def plot_speed_signals(self):
        """
        Function initially had the purpose of extracting the points where the speed change automatically
        """
        plt.figure()
        # for key, val in tqdm(self.datasets.items()):
        for key, val in tqdm(list(self.datasets.items())[0:1]):
            mat = loadmat(lms_path.joinpath(key), matlab_compatible=True, simplify_cells=True)
            speed_signal = mat["Signal_1"]["y_values"]["values"]
            speed_correction_factor = mat["Signal_1"]["y_values"]["quantity"]["unit_transformation"]["factor"]
            speed_signal = speed_signal * speed_correction_factor
            print(len(speed_signal))

            # Downsample the speed signal by cutting it into windows and taking the mean of each window
            window_size = 20000
            down_sampled_speed_signal = speed_signal[0:len(speed_signal) - len(speed_signal) % window_size].reshape(-1,
                                                                                                                    window_size).mean(
                axis=1)
            print(len(speed_signal))

            plt.plot(speed_signal, label=key)

            # # Also plot the derivative of the speed signal to find the change in speed
            # difference = np.diff(down_sampled_speed_signal)
            # # plt.plot(difference, label=val["mode"] + " 2nd derivative")
            #
            # # Get the first 4 peaks, ensuring a minimum distance
            # peaks, _ = find_peaks(difference, distance=30)
            # # Keep only the 4 peaks that correspond the the largest downsampled speed signal
            # peaks = peaks[np.argsort(down_sampled_speed_signal[peaks])[-5:]]
            #
            # # Plot the peak locations as vertical lines
            # # plt.vlines(peaks, ymin=difference.min(), ymax=difference.max(), alpha=0.3)
            # # plt.plot(peaks, difference[peaks], "x")
            #
            # # Get the coordinates of the peaks in the original signal
            # peaks = peaks*window_size
            #
            # # Plot the original signal with the peaks marked as vertical lines
            # plt.plot(speed_signal, label=val["mode"])
            # plt.vlines(peaks, ymin=speed_signal.min(), ymax=speed_signal.max(), alpha=0.3)
            #
            # total_extract_length = 4e5
            # for speedrange in range(4):
            #     start_point = peaks[speedrange] - 1e4 + total_extract_length
            #     to_keep = [start_point, start_point + total_extract_length]
            # # Show the rages to keep as vertical lines
            #     plt.vlines(to_keep, ymin=speed_signal.min(), ymax=speed_signal.max(), alpha=0.3, color="red")

        plt.legend()
        plt.show()


print("Adding LMS data to db")
LMS().add_all_to_db()

print("\n \n Creating noisy copies of the data")
create_noisy_copies_of_dataset.main("lms")

print("\n \n Computing features")
compute_features.main("lms")

print("\n \n Writing datasets to file")
get_data_from_db_and_save_to_file("lms")

# Usefull for diagnosing channels
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
