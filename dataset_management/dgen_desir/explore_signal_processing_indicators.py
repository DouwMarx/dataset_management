import pandas as pd
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from scipy import signal
import seaborn as sns

from dataset_management.dgen_desir.sigproc_pipeline import SimpleDemodulatedEnergySP
from dataset_management.dgen_desir.write_data_to_standard_structure import get_data



def make_plots_for_all_data():
    stationarities = ['stationary' , 'non_stationary']
    channels = [
        'acc1_X',  'acc1_Y',   'acc1_Z', 'acc2_X', 'acc2_Y', 'acc2_Z',
        'acc3_X', 'acc3_Y', 'acc3_Z', 'mic1', 'mic10', 'mic11',
        'mic2', 'mic3', 'mic4', 'mic5', 'mic6', 'mic7', 'mic8', 'mic9'
    ]
    performance_records = {}  # Will store {(channel, stationarity): {order_name: AUC}}
    segment_length = 20480
    rpm_measured_at_fan_directly = True
    fault_orders = {
        "Imb": 1,              # Imbalance (non-uniform flow)
        "Fan": 14,             # Fan blade passage (14 blades)
        "Fan+1": 15,           # Fan modulated by 1/rev
        "Fan-1": 13,           # Fan modulated by 1/rev
        "OGV": 40,             # OGV blade passage (40 vanes)
        "OGV+1": 41,           # OGV modulated by 1/rev (missing vanes)
        "OGV-1": 39,           # OGV modulated by 1/rev (missing vanes)
        "OGV-Fan+1": 27,       # Interaction: OGV-Fan (40-14) + 1/rev
        "OGV-Fan-1": 25,       # Interaction: OGV-Fan (40-14) - 1/rev
        "OGV+Fan+1": 55,       # Interaction: OGV+Fan (40+14) + 1/rev
        "OGV+Fan-1": 53        # Interaction: OGV+Fan (40+14) - 1/rev
    }
    if not rpm_measured_at_fan_directly:
        planetary_gearbox_ratio = 4400/14900 # From slide 19 in the presentation
        for key in fault_orders.keys():
            fault_orders[key] = fault_orders[key] * planetary_gearbox_ratio

    total_plots_to_make = len(channels) * len(stationarities)
    
    # Generate colors using the standard palette
    colors = plt.cm.rainbow(np.linspace(0, 1, len(fault_orders)))
    order_colors = dict(zip(fault_orders.keys(), colors))
    # Create one multi-page PDF
    with PdfPages(f"all_plots_segment_length_{segment_length}_rpm_measured_at_fan_directly_{rpm_measured_at_fan_directly}.pdf") as pdf:
        for channel in channels:
            for stationarity in stationarities:

                # Dictionary to store normal/faulty data for each order
                data_store = {
                    "Normal": {"indicators": {}, "example_segment": None},
                    "Faulty": {"indicators": {}, "example_segment": None}
                }

                # Dictionary to store ROC AUC values for each order
                roc_auc_values_for_different_orders = {}

                # Retrieve data/compute indicators for both Normal & Faulty
                fig, axs = plt.subplots(4, 2, figsize=(15, 16))
                fig.suptitle(f"Channel: {channel}, Stationarity: {stationarity}, Segment Length {segment_length}",
                             fontsize=16)

                for health_state, health_state_name in zip(['normal', 'faulty'], data_store.keys()):
                    data = get_data(health_state, stationarity, channel,
                                    segment_length=segment_length, decimate_factor=1)
                    segments = data['measurement_segments']
                    mean_rpm = data['mean_rpm']
                    fs = data['fs']

                    # Process each fault order
                    for order_name, order_value in fault_orders.items():
                        sp_method = SimpleDemodulatedEnergySP(target_order=order_value, fs=fs, mean_rpm=mean_rpm)
                        indicators = sp_method.get_test_statistic(segments)
                        data_store[health_state_name]["indicators"][order_name] = indicators

                    # Store example segment
                    data_store[health_state_name]["example_segment"] = segments[0]  # Example segment
                    data_store[health_state_name]["mean_rpm"] = data['mean_rpm']

                    # ───────────────────── Plot 1: Speed Profile ─────────────────────
                    axs[0, 0].plot(data['speed_rpm'], label=health_state_name)
                    axs[0, 0].set_title("Speed Profile")
                    axs[0, 0].set_xlabel("Segment Index")
                    axs[0, 0].set_ylabel("Speed (RPM)")
                    axs[0, 0].legend()

                # ───────────────────── Plot ROC Curves ─────────────────────
                for order_name, order_value in fault_orders.items():
                    y_true = np.concatenate([
                        np.zeros(len(data_store["Normal"]["indicators"][order_name])),
                        np.ones(len(data_store["Faulty"]["indicators"][order_name]))
                    ])
                    y_score = np.concatenate([
                        data_store["Normal"]["indicators"][order_name],
                        data_store["Faulty"]["indicators"][order_name]
                    ])
                    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
                    roc_auc = metrics.auc(fpr, tpr)
                    roc_auc_values_for_different_orders[order_name] = roc_auc

                    # Store AUC in nested dictionary
                    dataset_key = (channel, stationarity)
                    if dataset_key not in performance_records:
                        performance_records[dataset_key] = {}
                    performance_records[dataset_key][order_name] = roc_auc

                    axs[0, 1].plot(fpr, tpr, color=order_colors[order_name],
                                 label=f"{order_name} (Order {order_value}) AUC = {roc_auc:.3f}")

                axs[0, 1].plot([0, 1], [0, 1], 'k--')
                axs[0, 1].set_title("ROC Curves for Different Fault Orders")
                axs[0, 1].set_xlabel("False Positive Rate")
                axs[0, 1].set_ylabel("True Positive Rate")
                axs[0, 1].legend(fontsize='small')

                # ───────────────────── Plot 2: Indicator Trace (best performing order) ─────────────────────
                best_order = max(roc_auc_values_for_different_orders.items(), key=lambda x: x[1])[0]
                for health_state_name, data in data_store.items():
                    axs[1, 0].plot(data["indicators"][best_order],
                                 label=f"{health_state_name}",
                                 color=order_colors[best_order] if health_state_name == "Normal" else "gray")
                axs[1, 0].set_title(f"Indicator Trace - {best_order} (Order {fault_orders[best_order]}, Best AUC = {roc_auc_values_for_different_orders[best_order]:.3f})")
                axs[1, 0].set_xlabel("Segment Index")
                axs[1, 0].set_ylabel("Indicator Value")
                axs[1, 0].legend()

                # ───────────────────── Plot Indicator Histograms (best performing order) ─────────────────────
                best_order = max(roc_auc_values_for_different_orders.items(), key=lambda x: x[1])[0]
                best_auc = roc_auc_values_for_different_orders[best_order]

                axs[1, 1].hist(data_store["Normal"]["indicators"][best_order], bins=20, alpha=0.5,
                             label='Normal', color=order_colors[best_order])
                axs[1, 1].hist(data_store["Faulty"]["indicators"][best_order], bins=20, alpha=0.5,
                             label='Faulty', color='gray')
                axs[1, 1].set_title(f"Histogram of Indicators - {best_order} (Order {fault_orders[best_order]}, Best AUC = {best_auc:.3f})")
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

                # Set up plot parameters
                axs[2, 1].set_xlim(0, fs / 2)  # Real FFT frequencies up to Nyquist
                axs[2, 1].set_yscale('log')
                axs[2, 1].set_title("Example Segment (Frequency Domain)")
                axs[2, 1].set_xlabel("Frequency [Hz]")
                axs[2, 1].set_ylabel("Magnitude")

                # Calculate frequencies
                order_frequency = data_store["Faulty"]["mean_rpm"] / 60
                nyquist_freq = fs / 2
                max_order = int(nyquist_freq / order_frequency)

                # Plot vertical lines first (behind the data)
                # Integer orders as light gray lines
                for i in range(1, max_order + 1):
                    freq = i * order_frequency
                    if freq <= nyquist_freq:
                        axs[2, 1].axvline(freq, color='lightgray', linestyle='--', alpha=0.5,
                                         label=f"Order {i}" if i == 1 else "")

                # Use viridis colormap for fault orders
                distinct_colors = sns.color_palette("hls", len(fault_orders))
                fault_colors =  [distinct_colors[i] for i in range(len(fault_orders))]
                for (order_name, order_value), color in zip(fault_orders.items(), fault_colors):
                    fault_frequency = order_frequency * order_value
                    if fault_frequency <= nyquist_freq:
                        axs[2, 1].axvline(fault_frequency, color=color, linestyle='-', alpha=1,
                                         label=f"{order_name} (Order {order_value})")

                # Plot FFT data on top
                axs[2, 1].plot(freqs, normal_fft, label='Normal FFT', color='blue')
                axs[2, 1].plot(freqs, faulty_fft, label='Faulty FFT', color='red', alpha=0.7)
                axs[2, 1].legend(loc='upper right', fontsize='small')

                # ───────────────────── Plot 7: Welch PSD ─────────────────────
                normal_seg = data_store["Normal"]["example_segment"]
                faulty_seg = data_store["Faulty"]["example_segment"]

                # Calculate Welch PSD with segment length of 1000
                f_normal, Pxx_normal = signal.welch(normal_seg, fs=fs, nperseg=1000)
                f_faulty, Pxx_faulty = signal.welch(faulty_seg, fs=fs, nperseg=1000)

                # Set up plot parameters
                axs[3, 0].set_xlim(0, fs / 2)  # Frequencies up to Nyquist
                axs[3, 0].set_yscale('log')
                axs[3, 0].set_title("Example Welch Power Spectral Density")
                axs[3, 0].set_xlabel("Frequency [Hz]")
                axs[3, 0].set_ylabel("Power/Frequency [V^2/Hz]")

                # Plot vertical lines first (behind the data)
                # Integer orders as light gray lines
                for i in range(1, max_order + 1):
                    freq = i * order_frequency
                    if freq <= nyquist_freq:
                        axs[3, 0].axvline(freq, color='lightgray', linestyle='--', alpha=0.5,
                                         label=f"Order {i}" if i == 1 else "")

                # Use same viridis colors for fault orders
                for (order_name, order_value), color in zip(fault_orders.items(), fault_colors):
                    fault_frequency = order_frequency * order_value
                    if fault_frequency <= nyquist_freq:
                        axs[3, 0].axvline(fault_frequency, color=color, linestyle='-', alpha=1,
                                         label=f"{order_name} (Order {order_value})")

                # Plot PSD data on top
                axs[3, 0].plot(f_normal, Pxx_normal, label='Normal PSD', color='blue')
                axs[3, 0].plot(f_faulty, Pxx_faulty, label='Faulty PSD', color='red', alpha=0.7)
                axs[3, 0].legend(loc='upper right', fontsize='small')

                # Plot 8: Empty for now (right subplot in the new row)
                axs[3, 1].set_visible(False)  # Hide the empty subplot

                plt.tight_layout()
                # Save this figure to the PDF (adds a new page)
                pdf.savefig(fig)
                plt.close(fig)

                current_plot = channels.index(channel) * len(stationarities) + stationarities.index(stationarity) + 1
                print("Percentage done: {:.2f}%".format(100 * current_plot / total_plots_to_make))

        # Convert nested dictionary to DataFrame
        df_records = []
        for (channel, stationarity), order_aucs in performance_records.items():
            record = {
                "channel": channel,
                "stationarity": stationarity,
                "segment_length": segment_length
            }
            # Add AUC values for each order
            for order_name, auc in order_aucs.items():
                record[f"AUC_{order_name}"] = auc
            df_records.append(record)


        performance_df = pd.DataFrame(df_records)
        print("Performance Metrics:")
        print(performance_df)
        # Save the performance metrics to a CSV file
        performance_df.to_csv("performance_metrics.csv", index=False)

        # Of all numerical columns containing AUC values, find the min and max
        auc_columns = [col for col in performance_df.columns if col.startswith('AUC_')]
        df_min = performance_df[auc_columns].min().min()
        df_max = performance_df[auc_columns].max().max()

        norm = Normalize(vmin=df_min, vmax=df_max)
        cmap = plt.cm.YlOrRd  # Yellow-Orange-Red colormap

        # Create a color array for the AUC columns, initializing an empty array
        colors = np.empty_like(performance_df, dtype=object)

        # Populate the colors based on normalized AUC values, looping only through AUC columns
        for i in range(len(performance_df)):
            for j, col in enumerate(performance_df.columns):
                if col in auc_columns:
                    value = performance_df.at[i, col]
                    colors[i, j] = cmap(norm(value))
                else:
                    colors[i, j] = 'white'  # Default color for non-AUC columns

        performance_df = performance_df.round(2)
        # Render the dataframe as a page of the PDF
        fig = plt.figure(figsize=(12, 18))
        ax = fig.add_subplot(111)
        ax.axis('off')
        table = ax.table(cellText=performance_df.values,
                        colLabels=performance_df.columns,
                         cellColours= colors,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Save summary statistics of AUCs in table format
        auc_columns = [col for col in performance_df.columns if col.startswith('AUC_')]
        mean_aucs = {col.replace('AUC_', ''): performance_df[col].mean() for col in auc_columns}
        
        # Create summary DataFrame
        summary_df = pd.DataFrame([mean_aucs], index=['Mean AUC'])
        summary_df = summary_df.round(2)
        # Create figure for summary table
        fig = plt.figure(figsize=(12, 2))
        ax = fig.add_subplot(111)
        ax.axis('off')
        table = ax.table(cellText=summary_df.values,
                        rowLabels=summary_df.index,
                        colLabels=summary_df.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    make_plots_for_all_data()
