import pathlib

import numpy as np
import pandas as pd

from dataset_management.cwr_dataset.write_data_to_standard_structure import get_cwru_data_frame
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
processed_dir = pathlib.Path(__file__).parent.joinpath("processed_data")
if not processed_dir.exists():
    processed_dir.mkdir()

raw_folder = pathlib.Path(__file__).parent.joinpath("raw_data")
df = get_cwru_data_frame(min_average_events_per_rev=8,
                         overlap=0.5,
                         data_path=raw_folder
                         )

print("Dataset sizes:", df["Signals"].apply(lambda x:np.shape(x)[0]).unique())
print("Signal lengths:", df["Signals"].apply(lambda x:np.shape(x)[1]).unique())

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

measurement_locations = ['DE', 'FE']
unique_sampling_rates = df["Sampling Rate [kHz]"].unique()

for measurement_location in measurement_locations: # Make 4 separate datasets
    for sampling_rate in unique_sampling_rates:
        # Get the applicable datasets
        df_entries = []
        sub_data_sets = df[(df["Measurement Location"] == measurement_location) & (df["Sampling Rate [kHz]"] == sampling_rate)]

        # Do a separate split for each dataset to avoid data leakage
        for i,row in sub_data_sets.iterrows():
            signals_for_dataset = row["Signals"]
            fault_state = 0 if row["Fault Location"] == "NONE" else 1 # Healthy vs Faulty

            # Do splitting, considering that data has overlap
            n_signals = np.shape(signals_for_dataset)[0]
            n_signals = n_signals - 2  # Remove the signal between train and val and val and test to avoid leakage

            n_train = int(n_signals * train_ratio)
            n_val = int(n_signals * val_ratio)
            n_test = n_signals - n_train - n_val

            train_signals = signals_for_dataset[:n_train - 1, :, :]
            val_signals = signals_for_dataset[n_train:n_train + n_val - 1, :, :]
            test_signals = signals_for_dataset[n_train + n_val:, :, :]

            other_row_info = row.drop(["Signals"])
            other_row_info = other_row_info.to_dict()

            for set_of_signals, set_name in zip([train_signals, val_signals, test_signals], ["train", "val", "test"]):
                for signal in set_of_signals:
                    signal_dict = other_row_info.copy()
                    signal_dict["time series"] = signal.flatten()
                    signal_dict["label"] = fault_state
                    signal_dict["set"] = set_name
                    df_entries.append(signal_dict )

        # Create a dataframe
        dataset_df = pd.DataFrame(df_entries)

        # Save the data
        dataset_df.to_pickle(processed_dir.joinpath(f"{measurement_location}_{sampling_rate}_binary-all_fault_modes_speed_fault_location.pkl"))

