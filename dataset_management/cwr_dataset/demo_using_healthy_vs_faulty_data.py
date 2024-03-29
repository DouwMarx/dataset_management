import pathlib
import pandas as pd

processed_dir = pathlib.Path(__file__).parent.joinpath("processed_data")
df = pd.read_pickle(processed_dir.joinpath("DE_12_binary-all_fault_modes_speed_fault_location.pkl"))

"""
Processed data does healthy-faulty split, regardless of fault mode, operating condition and fault location.

There are 4 datasets, corresponding to the 2 sampling rates and 2 measurement locations. 

Each dataset has "set labels". This is because the data is split with overlap, and should not be split randomly.
The set labels avoid data leakage between the splits.

To make the task easier it could make sense to enforce that the measurement location and the fault location are the same.
 """

train_x = df[df["set"] == "train"]["time series"]
train_y = df[df["set"] == "train"]["label"]

val_x = df[df["set"] == "val"]["time series"]
val_y = df[df["set"] == "val"]["label"]

test_x = df[df["set"] == "test"]["time series"]
test_y = df[df["set"] == "test"]["label"]

print("Signal length: ", train_x[0].shape[0])

# Verify test/train/val split percentages
print("Train %: ", len(train_x)/len(df))
print("Val %: ", len(val_x)/len(df))
print("Test %: ", len(test_x)/len(df))