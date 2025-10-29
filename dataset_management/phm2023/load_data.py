import numpy as np
import pathlib
import joblib
import pandas as pd
import scipy
from joblib import Memory
import mat73

raw_data_path = pathlib.Path(__file__).parent.parent / "data" / "raw"
labeled_data_path = raw_data_path.joinpath(
    "Data_Challenge_PHM2023_training_data"
)  # Contains separate folders with data at different speed and load
preliminary_test_set_path = raw_data_path.joinpath(
    "Data_Challenge_PHM2023_test_data"
)  # Contains separate folders with data at different speed and load

# interim_data_path = pathlib.Path(__file__).parent.parent / 'data' / 'interim'
# processed_data_path = pathlib.Path(__file__).parent.parent / 'data' / 'processed'
interim_data_path = pathlib.Path(
    "/home/douwm/projects/phm2023_data_competition/data/interim"
)
processed_data_path = pathlib.Path(
    "/home/douwm/projects/phm2023_data_competition/data/processed"
)


cache2 = Memory(location=processed_data_path / "cache2", verbose=0)


@cache2.cache
def load_all_samples_from_folder(folder_path, labelled=True, file_type=".txt"):
    # Load all samples from a folder that is associated with a given health level
    # Extract the health level from the folder name

    print("loading samples from folder {}".format(folder_path))

    if labelled:
        health_level = int(
            folder_path.name.split("level_")[1][0]
        )  # (Number after word "level_")
    else:
        health_level = None

    samples = []
    for file in folder_path.iterdir():
        # Extract the operating speed (V), load (N) and sample number (before .txt) from the file name

        operating_speed = int(file.name.split("V")[1].split("_")[0])  # Between V and _

        try:
            if labelled:
                if file_type == ".txt":
                    sample_number = int(file.name.split(".txt")[0][-1])  # Before .txt
                elif file_type == ".mat":
                    sample_number = int(file.name.split(".mat")[0][-1])  # Before .mat
                else:
                    raise ValueError("file_type must be .txt or .mat")
                load = int(file.name.split("_")[1].split("N")[0])  # Between _ and N
            else:
                sample_number = int(file.name.split("_")[0])  # First numbers before _
                # Between second _ and N
                load = int(file.name.split("_")[2].split("N")[0])  # Between _ and N
        except:
            raise ValueError(
                "The file name {} is not in the expected format for labelled = {}".format(
                    file.name, labelled
                )
            )

        fname = file.name

        if file_type == ".txt":
            data = np.loadtxt(file)
        elif file_type == ".mat":
            # data = scipy.io.loadmat(file)
            data = (
                np.abs(mat73.loadmat(file)["csc"]) + 1e-10
            )  # Make sure data is strictly positive
        else:
            raise ValueError("file_type must be .txt or .mat")

        sample_dict = {
            "health_level": health_level,
            "speed": operating_speed,
            "load": load,
            "sample_number": sample_number,
            "data": data,
            "file_name": fname,
            "labelled": labelled,
        }
        samples.append(sample_dict)
    return samples
