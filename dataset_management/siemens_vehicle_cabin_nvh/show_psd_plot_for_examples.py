import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import welch

data_dir = pathlib.Path("/home/douwm/data/siemens_isma_2024")
image_dir = pathlib.Path("/home/douwm/projects/PhD/resources/images_and_descriptions/images/plots/signal_processing_jacobian")

n_vehicles_per_mode = "all"
segment_length = 4096

for vehicle in ["Mondeo",  "Vectra"]:
    data_map = np.lib.format.open_memmap(data_dir.joinpath(vehicle + "_sounds.npy"), mode='r+')
    meta_data = pd.read_excel(data_dir.joinpath(vehicle + "_simdata.xlsx"))

    saved_meta_data = {"sampling_frequency": 44100,
                       "n_vehicles_per_mode": n_vehicles_per_mode,
                       "segment_length": segment_length,
                       "vehicle": vehicle
                       }

    normal_query = (meta_data['ComputeLeakageBin'] == 0) & (meta_data['ComputeWhistleBin'] == 0)
    whistle_query = (meta_data['ComputeLeakageBin'] == 0) & (meta_data['ComputeWhistleBin'] == 1)
    leakage_query = (meta_data['ComputeLeakageBin'] == 1) & (meta_data['ComputeWhistleBin'] == 0)

    normal_meta_data = meta_data[normal_query]
    whistle_meta_data = meta_data[whistle_query]
    leakage_meta_data = meta_data[leakage_query]
    normal_indexes = normal_meta_data.index
    whistle_indexes = whistle_meta_data.index
    leakage_indexes = leakage_meta_data.index

    fig_psd = plt.figure("psd")
    added_labels = []
    # for mode_name, mode_data,mode_color in zip(["whistle", "leakage", "normal"], [whistle_meta_data, leakage_meta_data, normal_meta_data],["red", "blue", "green"]):
    for mode_name, mode_data,mode_color in zip(["Whistle fault", "Normal"], [whistle_meta_data, normal_meta_data],["red", "green"]):
        for example_index in np.random.choice(mode_data.index, 5):
            # Full psd (averaging)
            plt.figure("psd")
            sig_full =  data_map[example_index]
            f, Pxx = welch(sig_full, fs=44100, nperseg=1024)
            if mode_name not in added_labels:
                plt.semilogy(f, Pxx, label=mode_name, color = mode_color, alpha=0.5)
                added_labels.append(mode_name)
            else:
                plt.semilogy(f, Pxx, color = mode_color, alpha=0.5)
    plt.title(f"Vehicle: {vehicle}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power spectral density")
    plt.legend()
    plt.savefig(image_dir.joinpath(f"psd_{vehicle}.pdf"))

    plt.show()
