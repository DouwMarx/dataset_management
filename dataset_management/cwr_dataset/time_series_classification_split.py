import pathlib
import numpy as np
import pandas as pd

from dataset_management.cwr_dataset.write_data_to_standard_structure import get_cwru_data_frame
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
processed_dir = pathlib.Path(__file__).parent.joinpath("processed_data")
classification_task_dir = processed_dir.joinpath("classification_task_data")

if not classification_task_dir.exists():
    classification_task_dir.mkdir()

raw_folder = pathlib.Path(__file__).parent.joinpath("raw_data")
df = get_cwru_data_frame(min_average_events_per_rev=10,
                         overlap=0.5,
                         data_path=raw_folder
                         )


train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

sampling_rate = "12"
measurement_location = "DE"
fault_location = "DE"

# Get the applicable datasets
df_entries = []
sub_data_sets = df[
    (df["Measurement Location"] == measurement_location) &\
    (df["Sampling Rate [kHz]"] == sampling_rate) &\
    (df["Fault Location"].isin(["NONE", fault_location])) # Only faults that occur at the measurement location
]

print("Dataset sizes:", df["Signals"].apply(lambda x:np.shape(x)[0]).unique())
print("Signal lengths:", df["Signals"].apply(lambda x:np.shape(x)[-1]).unique())
# Remove 'File Number', 'Fault Frequency'
sub_data_sets = sub_data_sets.drop(["File Number", "Fault Frequency"], axis=1)

# Remove Fault mode Outer Race Fault: Orthogonal and Outer Race: Opposite
sub_data_sets = sub_data_sets[~sub_data_sets["Fault Mode"].isin(["Outer Race Fault: Orthogonal", "Outer Race: Opposite "])]
# Check sub_data_sets to understand which fault modes are present

fault_modes_present = sub_data_sets["Fault Mode"].unique()
label_mapping = {fault_mode: i for i, fault_mode in enumerate(fault_modes_present)}
print("Label mapping: ", label_mapping)

# Do a separate split for each sub-dataset (operating condition/fault mode combination) to avoid data leakage due to overlap
for i, row in sub_data_sets.iterrows():
    signals_for_dataset = row["Signals"]

    # Do splitting, considering that data has overlap
    n_signals = np.shape(signals_for_dataset)[0]
    n_signals = n_signals - 2  # Remove the signal between train and val and val and test to avoid leakage (50% overlap)

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
            signal_dict["label"] =  int(label_mapping[row["Fault Mode"]])
            signal_dict["set"] = set_name
            df_entries.append(signal_dict )

# Create a dataframe
dataset_df = pd.DataFrame(df_entries)
# Set the label type to int
dataset_df["label"] = dataset_df["label"].astype(int)
# Shuffle the rows so you cant draw conclusions based on the order of the train data
dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)

# For the test, train and validation sets, write a separate folder containing
# 1) The signals as a csv where each row is a signal
# 2) A csv called labels containing only the labels
# 3) A csv called metadata_and_labels containing the metadata and the labels
for set_name in ["train", "val", "test"]:
    set_df = dataset_df[dataset_df["set"] == set_name]
    set_df = set_df.drop(["set"], axis=1)

    # Write the signals
    signals = np.stack(set_df["time series"].apply(lambda x: np.array(x)).values)
    np.savetxt(classification_task_dir.joinpath(f"{set_name}_signals.csv"), signals, delimiter=",")

    # Write the labels
    labels = set_df["label"].values
    np.savetxt(classification_task_dir.joinpath(f"{set_name}_labels.csv"), labels, delimiter=",", fmt="%d")

    # Write the metadata and labels
    # For the test set, the fault mode and labels are excluded
    if set_name == "test":
        set_df = set_df.drop(["Fault Location", "Fault Width [mm]", "Fault Mode", "label"], axis=1)

    metadata_and_labels = set_df.drop(["time series"], axis=1)
    metadata_and_labels.to_csv(classification_task_dir.joinpath(f"{set_name}_metadata.csv"), index=False)
