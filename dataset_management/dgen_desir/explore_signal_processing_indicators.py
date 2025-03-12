import pathlib

import sklearn.metrics as metrics
import numpy as np
import scipy.io as sio
import scipy.interpolate as interp
import matplotlib.pyplot as plt

from dataset_management.dgen_desir.sigproc_pipeline import SP, SimpleDemodulatedEnergySP


def get_data(condition,
             stationarity,
             channel,
             segment_length=1000,
             dataset_path="/home/douwm/data/DGEN380_turbofan/",
             decimate_factor=1,
             ):
    # Construct file paths based on condition and stationarity
    stationarity_suffix = '1' if stationarity == 'stationary' else '2'
    config_num = '2' if condition == 'normal' else '4' # config 2 for normal, 4 for faulty

    dataset_path = pathlib.Path(dataset_path)

    data_path = f"desir_ii_configuration_{config_num}_{condition}_{stationarity_suffix}.mat"
    performance_path = f"desir_ii_config_{config_num}_perfo_{condition}_{stationarity_suffix}.mat"

    data_path = dataset_path.joinpath(data_path)
    performance_path = dataset_path.joinpath(performance_path)

    # Load data
    data = sio.loadmat(data_path.__str__())

    performance = sio.loadmat(performance_path.__str__())

    # print("Channels available in meusurement data: ", data.keys())
    # print("Channels available in performance data: ", performance.keys())

    full_measurement = data[channel].astype(float).flatten()
    time_at_speed = performance['t'].flatten()
    speed_rpm = performance['NH'].flatten()

    if decimate_factor > 1:
        from scipy.signal import decimate
        full_measurement = decimate(full_measurement, decimate_factor)
        fs = 40960 // decimate_factor
    else:
        fs = 40960

    measurement_segments = cut_into_non_overlapping_segments(full_measurement, segment_length)

    return {'full_measurement': full_measurement,
            'time_at_speed': time_at_speed,
             'speed_rpm': speed_rpm,
              'measurement_segments': measurement_segments,
            # 'speed_segments_rpm': speed_segments_rpm
            'mean_rpm': np.mean(speed_rpm),
            "fs": fs
            }

def cut_into_non_overlapping_segments(signal, segment_length=5000):
    """
    Cuts a signal into non-overlapping segments of length segment_length
    :param signal: The signal to be cut
    :return: Array of segments
    """
    num_segments = len(signal) // segment_length
    segments = signal[:num_segments * segment_length].reshape(num_segments, segment_length)
    return segments



# if __name__ == "__main__":
#     stationarities = ['stationary', 'non_stationary']
#     domains = ['time', 'angular']
#     channels = ['acc1_X', 'acc1_Y', 'acc1_Z', 'acc2_X', 'acc2_Y', 'acc2_Z', 'acc3_X', 'acc3_Y', 'acc3_Z', 'mic1', 'mic10', 'mic11', 'mic2', 'mic3', 'mic4', 'mic5', 'mic6', 'mic7', 'mic8', 'mic9']
#
#     for channel in channels:
#         for stationarity in stationarities:
#             indicator_store = {"Normal": None, "Faulty": None}
#
#             # Make a plot with 4 subplots
#             # On each subplot, both the normal and faulty conditions are plotted as different traces
#             #
#             # The first row of subplots shows: The speed profile, The ROC curve for the indicator values
#             # The second row of subplots shows: The indicator values for different segments over time, Histograms of the indicator values
#
#             fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#
#             for health_state,health_state_name in zip(['normal', 'faulty'], list(indicator_store.keys())):
#                 data = get_data(health_state, stationarity, channel, segment_length=5000, decimate_factor=2)
#                 segments = data['measurement_segments']
#                 mean_rpm = data['mean_rpm']
#                 fs = data['fs']
#
#                 sp_method = SimpleDemodulatedEnergySP(target_order=14, fs = fs, mean_rpm=mean_rpm)
#                 indicators = sp_method.get_test_statistic(segments)
#
#                 indicator_store[health_state_name] = indicators
#
#                 axs = axs.flatten()
#                 axs[0].plot(data['speed_rpm'], label=health_state_name)
#                 axs[0].set_title("Speed profile")
#
#             # Make ROC curve
#             y_true = np.concatenate([np.zeros(len(indicator_store["Normal"])), np.ones(len(indicator_store["Faulty"]))])
#             y_score = np.concatenate([indicator_store["Normal"], indicator_store["Faulty"]])
#             fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
#             fig.show()

if __name__ == "__main__":
    stationarities = ['stationary', 'non_stationary']
    channels = [
        'acc1_X', 'acc1_Y', 'acc1_Z', 'acc2_X', 'acc2_Y', 'acc2_Z',
        'acc3_X', 'acc3_Y', 'acc3_Z', 'mic1', 'mic10', 'mic11',
        'mic2', 'mic3', 'mic4', 'mic5', 'mic6', 'mic7', 'mic8', 'mic9'
    ]

    for channel in channels:
        for stationarity in stationarities:
            data_store = {
                "Normal": {"indicators": None, "example_segment": None},
                "Faulty": {"indicators": None, "example_segment": None}
            }

            fig, axs = plt.subplots(3, 2, figsize=(15, 12))

            for health_state, health_state_name in zip(['normal', 'faulty'], data_store.keys()):
                data = get_data(health_state, stationarity, channel, segment_length=5000, decimate_factor=2)
                segments = data['measurement_segments']
                mean_rpm = data['mean_rpm']
                fs = data['fs']

                sp_method = SimpleDemodulatedEnergySP(target_order=14, fs=fs, mean_rpm=mean_rpm)
                indicators = sp_method.get_test_statistic(segments)

                # Store indicators and an example segment
                data_store[health_state_name]["indicators"] = indicators
                data_store[health_state_name]["example_segment"] = segments[0]  # Example segment

                axs[0, 0].plot(data['speed_rpm'], label=health_state_name)
                axs[0, 0].set_title("Speed Profile")
                axs[0, 0].legend()

                # Indicators Plot
                axs[1, 0].plot(indicators, label=f"{health_state_name} Indicator")
                axs[1, 0].set_title("Indicator Trace")
                axs[1, 0].legend()

            # ROC Curve
            y_true = np.concatenate([
                np.zeros(len(data_store["Normal"]["indicators"])),
                np.ones(len(data_store["Faulty"]["indicators"]))
            ])
            y_score = np.concatenate([
                data_store["Normal"]["indicators"],
                data_store["Faulty"]["indicators"]
            ])
            fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
            axs[0, 1].plot(fpr, tpr)
            axs[0, 1].plot([0, 1], [0, 1], 'k--')
            axs[0, 1].set_title("ROC Curve")
            axs[0, 1].set_xlabel("False Positive Rate")
            axs[0, 1].set_ylabel("True Positive Rate")

            # Indicator Histograms
            axs[1, 1].hist(data_store["Normal"]["indicators"], bins=20, alpha=0.5, label='Normal')
            axs[1, 1].hist(data_store["Faulty"]["indicators"], bins=20, alpha=0.5, label='Faulty')
            axs[1, 1].set_title("Histogram of Indicators")
            axs[1, 1].legend()

            # Time Domain Example Segments
            time = np.arange(len(data_store["Normal"]["example_segment"])) / fs
            axs[2, 0].plot(time, data_store["Normal"]["example_segment"], label='Normal Segment')
            axs[2, 0].plot(time, data_store["Faulty"]["example_segment"], label='Faulty Segment', alpha=0.7)
            axs[2, 0].set_title("Example Segment (Time Domain)")
            axs[2, 0].legend()

            # Frequency Domain Example Segments
            freqs = np.fft.fftfreq(len(data_store["Normal"]["example_segment"]), 1 / fs)
            normal_fft = np.abs(np.fft.fft(data_store["Normal"]["example_segment"]))
            faulty_fft = np.abs(np.fft.fft(data_store["Faulty"]["example_segment"]))

            axs[2, 1].plot(freqs, normal_fft, label='Normal FFT')
            axs[2, 1].plot(freqs, faulty_fft, label='Faulty FFT', alpha=0.7)
            axs[2, 1].set_xlim(0, fs / 2)  # Show only positive frequencies
            axs[2, 1].set_title("Example Segment (Frequency Domain)")
            axs[2, 1].legend()

            plt.tight_layout()
            plt.show()








