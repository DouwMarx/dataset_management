import pathlib
import numpy as np
import pandas as pd
from dataset_management.ultils.write_data_in_standard_format import export_data_to_file_structure
from file_definitions import cwr_path, biased_anomaly_detection_path
import matplotlib.pyplot as plt
from scipy.signal import welch

data_dir = pathlib.Path("/home/douwm/data/siemens_isma_2024")
n_vehicles_per_mode = "all"
segment_length = 4096  #  2048 #  8192 # 4096  # 1024 # 4096 # 2048 # 1024 # 4096# 1024 # 2048

for vehicle in ["Mondeo",  "Vectra"]:
    data_map = np.lib.format.open_memmap(data_dir.joinpath(vehicle + "_sounds.npy"), mode='r+')
    meta_data = pd.read_excel(data_dir.joinpath(vehicle + "_simdata.xlsx"))

    saved_meta_data = {"sampling_frequency": 44100,
                         "n_vehicles_per_mode": n_vehicles_per_mode,
                         "segment_length": segment_length,
                         "vehicle": vehicle
                         }

    # indexes_to_keep_per_mode = np.arange(n_vehicles_per_mode) # Keep the first 10 samples of each mode for now
    # Randomly select samples from each mode

    if n_vehicles_per_mode == "all":
        n_vehicles_per_mode = len(meta_data)
    else:
        n_vehicles_per_mode = min(n_vehicles_per_mode, len(meta_data))

    # Only use the first 1000 lines of the metadata since there is a difference in dataset i.e. vectra_1_1 and vectra_1
    # meta_data = meta_data.iloc[:1000]

    normal_query = (meta_data['ComputeLeakageBin'] == 0) & (meta_data['ComputeWhistleBin'] == 0)
    whistle_query = (meta_data['ComputeLeakageBin'] == 0) & (meta_data['ComputeWhistleBin'] == 1)
    leakage_query = (meta_data['ComputeLeakageBin'] == 1) & (meta_data['ComputeWhistleBin'] == 0)

    normal_meta_data = meta_data[normal_query]
    whistle_meta_data = meta_data[whistle_query]
    leakage_meta_data = meta_data[leakage_query]

    # randomly select n_vehicles_per_mode samples from each mode
    # normal_indexes = np.random.choice(normal_meta_data.index, n_vehicles_per_mode, replace=False)
    # whistle_indexes = np.random.choice(whistle_meta_data.index, n_vehicles_per_mode, replace=False)
    # leakage_indexes = np.random.choice(leakage_meta_data.index, n_vehicles_per_mode, replace=False)
    normal_indexes = normal_meta_data.index
    whistle_indexes = whistle_meta_data.index
    leakage_indexes = leakage_meta_data.index

    number_of_samples_per_mode = [len(normal_indexes), len(whistle_indexes), len(leakage_indexes)]
    print("Number of samples")
    print(number_of_samples_per_mode)

    normal_data = data_map[normal_indexes][:, :segment_length] # Using first 1k  samples in case the might might have been issues due to non-periodic signals in sim
    whistle_data = data_map[whistle_indexes][:, :segment_length]
    leakage_data = data_map[leakage_indexes][:, :segment_length]

    # Save the whistle_data for faster prototyping
    np.save(data_dir.joinpath(vehicle + "_normal_sounds.npy"), normal_data)
    np.save(data_dir.joinpath(vehicle + "_whistle_sounds.npy"), whistle_data)


    print(vehicle)

    # Add a channel dimension
    normal_data = np.expand_dims(normal_data, axis=1)
    whistle_data = np.expand_dims(whistle_data, axis=1)
    leakage_data = np.expand_dims(leakage_data, axis=1)

    # Compute the spectral flatness of the wistle data and print mean and std
    normal_spectrum = np.fft.rfft(normal_data, axis=-1)
    whistle_spectrum = np.fft.rfft(whistle_data, axis=-1)

    freqs = np.fft.rfftfreq(segment_length, d=1/44100)
    low_cut = 1000
    high_cut = 20000
    low_cut_index = np.argmin(np.abs(freqs - low_cut))
    high_cut_index = np.argmin(np.abs(freqs - high_cut))

    normal_amplitude_spectrum_mean = np.mean(np.abs(normal_spectrum), axis=0, keepdims=True)

    # Normalize
    normal_spectrum = normal_spectrum / normal_amplitude_spectrum_mean
    whistle_spectrum = whistle_spectrum / normal_amplitude_spectrum_mean

    normal_spectrum = np.abs(normal_spectrum)**2
    whistle_spectrum = np.abs(whistle_spectrum)**2

    normal_spectrum = normal_spectrum[:,:, low_cut_index:high_cut_index]
    whistle_spectrum = whistle_spectrum[:,:, low_cut_index:high_cut_index]

    normal_spectral_flatness = np.exp(np.mean(np.log(normal_spectrum), axis=-1)) / np.mean(normal_spectrum, axis=2)
    whistle_spectral_flatness = np.exp(np.mean(np.log(whistle_spectrum), axis=-1)) / np.mean(whistle_spectrum, axis=2)

    print(f"Normal spectral flatness: {np.mean(normal_spectral_flatness):.2f} +/- {np.std(normal_spectral_flatness):.2f}")
    print(f"Whistle spectral flatness: {np.mean(whistle_spectral_flatness):.2f} +/- {np.std(whistle_spectral_flatness):.2f}")

    faulty_data_dict = {
                        "whistle": whistle_data,
                        # "leakage": leakage_data
                        }

    show_plots = False
    n_examples = 10
    if show_plots:
        fig_psd = plt.figure("psd")
        for mode_name, mode_data,mode_color in zip(["whistle", "leakage", "normal"], [whistle_meta_data, leakage_meta_data, normal_meta_data],["red", "blue", "green"]):
            for example_index in np.random.choice(mode_data.index, 5):

                # Full psd (averaging)
                plt.figure("psd")
                sig_full =  data_map[example_index]
                f, Pxx = welch(sig_full, fs=44100, nperseg=1024)
                plt.semilogy(f, Pxx, label=mode_name, color = mode_color, alpha=0.5)
        plt.title(f"Power spectral density examples for {vehicle}")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power spectral density")
        plt.legend()

        for freq in [20, 2000, 5000, 20000]:
            plt.axvline(freq, color='r', linestyle='--')

        plt.title(f"Power spectral density examples for {vehicle}")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power spectral density")
        plt.legend()
        plt.show()

        # Plot examples of standardized samples
        healthy_fft = np.fft.rfft(normal_data, axis=-1)
        healthy_std = np.std(healthy_fft, axis=0, keepdims=True)

        fig_ps = plt.figure("standardized")
        for mode_name, mode_data, mode_color in zip(["whistle", "leakage", "normal"], [whistle_data, leakage_data, normal_data],["red", "blue", "green"]):
            for example_index in np.random.choice(range(np.min(number_of_samples_per_mode)), n_examples):
                sample_fft = np.abs(np.fft.rfft(mode_data[example_index], axis=-1))**2
                freqs = np.fft.rfftfreq(segment_length, d=1/44100)
                sample_fft = sample_fft/healthy_std
                plt.semilogy(freqs, sample_fft.flatten(), label=mode_name, color=mode_color, alpha=0.5)
        plt.title(f"Standardized FFT examples for {vehicle}")
        plt.legend()
        plt.show()

    export_data_to_file_structure(dataset_name="siemens_vc_nvh_{}".format(vehicle.lower()),
                                  healthy_data=normal_data,
                                  faulty_data_dict=faulty_data_dict,
                                  export_path=biased_anomaly_detection_path,
                                  metadata=saved_meta_data
                                  )
