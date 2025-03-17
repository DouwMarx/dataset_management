import pathlib
import pandas as pd
import sklearn.metrics as metrics
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from dataset_management.dgen_desir.sigproc_pipeline import SP, SimpleDemodulatedEnergySP
from dataset_management.ultils.write_data_in_standard_format import export_data_to_file_structure
from file_definitions import biased_anomaly_detection_path


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

def cut_into_non_overlapping_segments(signal, segment_length=1000):
    """
    Cuts a signal into non-overlapping segments of length segment_length
    :param signal: The signal to be cut
    :return: Array of segments
    """
    num_segments = len(signal) // segment_length
    segments = signal[:num_segments * segment_length].reshape(num_segments, segment_length)
    return segments

def make_plots_for_all_data():
    stationarities = ['stationary', 'non_stationary']
    channels = [
        'acc1_X', 'acc1_Y', 'acc1_Z', 'acc2_X', 'acc2_Y', 'acc2_Z',
        'acc3_X', 'acc3_Y', 'acc3_Z', 'mic1', 'mic10', 'mic11',
        'mic2', 'mic3', 'mic4', 'mic5', 'mic6', 'mic7', 'mic8', 'mic9'
    ]
    # stationarities = ['stationary']
    # channels = [
    #          'mic2', 'mic3'
    # ]

    performance_records = []  # Will store {channel, stationarity, AUC}
    segment_length = 2000
    target_order = 14

    total_plots_to_make = len(channels) * len(stationarities)
    # Create one multi-page PDF
    with PdfPages("all_plots.pdf") as pdf:
        for channel in channels:
            for stationarity in stationarities:

                # Dictionary to store normal/faulty data
                data_store = {
                    "Normal": {"indicators": None, "example_segment": None},
                    "Faulty": {"indicators": None, "example_segment": None}
                }

                # Retrieve data/compute indicators for both Normal & Faulty
                fig, axs = plt.subplots(3, 2, figsize=(15, 12))
                fig.suptitle(f"Channel: {channel}, Stationarity: {stationarity}, Segment Length {segment_length}",
                             fontsize=16)

                for health_state, health_state_name in zip(['normal', 'faulty'], data_store.keys()):
                    data = get_data(health_state, stationarity, channel,
                                    segment_length=segment_length, decimate_factor=1)
                    segments = data['measurement_segments']
                    mean_rpm = data['mean_rpm']
                    fs = data['fs']

                    sp_method = SimpleDemodulatedEnergySP(target_order=target_order, fs=fs, mean_rpm=mean_rpm)
                    indicators = sp_method.get_test_statistic(segments)

                    # Store indicators and an example segment
                    data_store[health_state_name]["indicators"] = indicators
                    data_store[health_state_name]["example_segment"] = segments[0]  # Example segment

                    # ───────────────────── Plot 1: Speed Profile ─────────────────────
                    axs[0, 0].plot(data['speed_rpm'], label=health_state_name)
                    axs[0, 0].set_title("Speed Profile")
                    axs[0, 0].set_xlabel("Segment Index")
                    axs[0, 0].set_ylabel("Speed (RPM)")
                    axs[0, 0].legend()

                    # ───────────────────── Plot 2: Indicator Trace ─────────────────────
                    axs[1, 0].plot(indicators, label=f"{health_state_name}")
                    axs[1, 0].set_title("Indicator Trace")
                    axs[1, 0].set_xlabel("Segment Index")
                    axs[1, 0].set_ylabel("Indicator Value")
                    axs[1, 0].legend()

                # ───────────────────── Plot 3: ROC Curve ─────────────────────
                y_true = np.concatenate([
                    np.zeros(len(data_store["Normal"]["indicators"])),
                    np.ones(len(data_store["Faulty"]["indicators"]))
                ])
                y_score = np.concatenate([
                    data_store["Normal"]["indicators"],
                    data_store["Faulty"]["indicators"]
                ])
                fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
                roc_auc = metrics.auc(fpr, tpr)

                # Save the AUC for later dataframe creation
                performance_records.append({
                    "channel": channel,
                    "stationarity": stationarity,
                    "segment_length": segment_length,
                    "target_order": target_order,
                    "AUC": roc_auc
                })

                axs[0, 1].plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
                axs[0, 1].plot([0, 1], [0, 1], 'k--')
                axs[0, 1].set_title("ROC Curve")
                axs[0, 1].set_xlabel("False Positive Rate")
                axs[0, 1].set_ylabel("True Positive Rate")
                axs[0, 1].legend()

                # ───────────────────── Plot 4: Indicator Histograms ─────────────────────
                axs[1, 1].hist(data_store["Normal"]["indicators"], bins=20, alpha=0.5, label='Normal')
                axs[1, 1].hist(data_store["Faulty"]["indicators"], bins=20, alpha=0.5, label='Faulty')
                axs[1, 1].set_title("Histogram of Indicators")
                axs[1, 1].set_xlabel("Indicator Value")
                axs[1, 1].set_ylabel("Count")
                axs[1, 1].legend()

                # ───────────────────── Plot 5: Time Domain Example ─────────────────────
                time = np.arange(len(data_store["Normal"]["example_segment"])) / fs
                axs[2, 0].plot(time, data_store["Normal"]["example_segment"], label='Normal Segment')
                axs[2, 0].plot(time, data_store["Faulty"]["example_segment"], label='Faulty Segment', alpha=0.7)
                axs[2, 0].set_title("Example Segment (Time Domain)")
                axs[2, 0].set_xlabel("Time [s]")
                axs[2, 0].set_ylabel("Amplitude")
                axs[2, 0].legend()

                # ───────────────────── Plot 6: Frequency Domain Example ─────────────────────
                normal_seg = data_store["Normal"]["example_segment"]
                faulty_seg = data_store["Faulty"]["example_segment"]

                freqs = np.fft.rfftfreq(len(normal_seg), 1 / fs)
                normal_fft = np.abs(np.fft.rfft(normal_seg))
                faulty_fft = np.abs(np.fft.rfft(faulty_seg))

                axs[2, 1].plot(freqs, normal_fft, label='Normal FFT')
                axs[2, 1].plot(freqs, faulty_fft, label='Faulty FFT', alpha=0.7)
                axs[2, 1].set_xlim(0, fs / 2)  # Real FFT frequencies up to Nyquist
                axs[2, 1].set_title("Example Segment (Frequency Domain)")
                axs[2, 1].set_xlabel("Frequency [Hz]")
                axs[2, 1].set_ylabel("Magnitude")

                # Plot a vertical line at the target order
                target_frequency = data['mean_rpm'] / 60 * target_order  # (rev/min) / (60 s/min) * (events/rev) = events/s
                axs[2, 1].axvline(target_frequency, color='r', linestyle='--', label=f"Target Order {target_order}")

                axs[2, 1].legend()

                plt.tight_layout()
                # Save this figure to the PDF (adds a new page)
                pdf.savefig(fig)
                plt.close(fig)

                print("Percentage done: {:.2f}%".format(100 * len(performance_records) / total_plots_to_make))

        # At the end, create a DataFrame with the performance records and print it
        performance_df = pd.DataFrame(performance_records)
        print("Performance Metrics:")
        print(performance_df)
        # Save the performance metrics to a CSV file
        performance_df.to_csv("performance_metrics.csv", index=False)

        # Render the dataframe as the last page of the PDF
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.table(cellText=performance_df.values, colLabels=performance_df.columns, cellLoc='center', loc='center')
        # Make sure the table does not overflow the page
        fig.tight_layout()
        pdf.savefig(fig)

if __name__ == "__main__":
    # make_plots_for_all_data()

    channel = "acc1_X"
    stationarity = "stationary"
    segment_length = 1000

    normal_data = get_data("normal", stationarity, channel, segment_length=segment_length, decimate_factor=1)
    faulty_data = get_data("faulty", stationarity, channel, segment_length=segment_length, decimate_factor=1)


    normal_segments = normal_data['measurement_segments']
    faulty_segments = faulty_data['measurement_segments']

    # add a channel dimension (batch, dim) -> (batch, 1, dim)
    normal_segments = np.expand_dims(normal_segments, axis=1)
    faulty_segments = np.expand_dims(faulty_segments, axis=1)

    faulty_data = {
        "ogv_blades": faulty_segments
    }

    meta_data = {
        "sampling_frequency": normal_data['fs'],
        "segment_length": segment_length
    }

    export_data_to_file_structure(dataset_name= "DGEN380_" + channel + "_" + stationarity,
                                  healthy_data=normal_segments,
                                  faulty_data_dict=faulty_data,
                                  export_path=biased_anomaly_detection_path,
                                  metadata=meta_data
                                  )
