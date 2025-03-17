import numpy as np
import pathlib
import joblib
import pandas as pd
import scipy
from joblib import Memory
import mat73

raw_data_path = pathlib.Path(__file__).parent.parent / 'data' / 'raw'
labeled_data_path = raw_data_path.joinpath(
    "Data_Challenge_PHM2023_training_data")  # Contains separate folders with data at different speed and load
preliminary_test_set_path = raw_data_path.joinpath(
    "Data_Challenge_PHM2023_test_data")  # Contains separate folders with data at different speed and load

# interim_data_path = pathlib.Path(__file__).parent.parent / 'data' / 'interim'
# processed_data_path = pathlib.Path(__file__).parent.parent / 'data' / 'processed'
interim_data_path = pathlib.Path("/home/douwm/projects/phm2023_data_competition/data/interim")
processed_data_path = pathlib.Path("/home/douwm/projects/phm2023_data_competition/data/processed")


cache2 = Memory(location=processed_data_path/ 'cache2', verbose=0)
@cache2.cache
def load_all_samples_from_folder(folder_path, labelled=True,file_type=".txt"):
    # Load all samples from a folder that is associated with a given health level
    # Extract the health level from the folder name

    print("loading samples from folder {}".format(folder_path))

    if labelled:
        health_level = int(folder_path.name.split("level_")[1][0])  # (Number after word "level_")
    else:
        health_level = None

    samples = []
    for file in folder_path.iterdir():
        # Extract the operating speed (V), load (N) and sample number (before .txt) from the file name

        operating_speed = int(file.name.split("V")[1].split("_")[0])  # Between V and _

        try:
            if labelled:
                if file_type == ".txt":
                    sample_number = int(file.name.split(".txt")[0][-1]) # Before .txt
                elif file_type == ".mat":
                    sample_number = int(file.name.split(".mat")[0][-1]) # Before .mat
                else:
                    raise ValueError("file_type must be .txt or .mat")
                load = int(file.name.split("_")[1].split("N")[0])  # Between _ and N
            else:
                sample_number = int(file.name.split("_")[0])  # First numbers before _
                # Between second _ and N
                load = int(file.name.split("_")[2].split("N")[0])  # Between _ and N
        except:
            raise ValueError(
                "The file name {} is not in the expected format for labelled = {}".format(file.name, labelled))

        fname = file.name

        if file_type == ".txt":
            data = np.loadtxt(file)
        elif file_type == ".mat":
            # data = scipy.io.loadmat(file)
            data = np.abs(mat73.loadmat(file)["csc"]) + 1e-10 # Make sure data is strictly positive
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


def write_interim_to_same_directory_structure(raw_data_path,
                                              function_to_apply,
                                              name_of_interim_operation="processed",
                                              parallel=True,
                                              return_dataframe=False,
                                              write_to_interim=True,
                                               file_type = ".txt"
                                              ):
    # Duplicate the directory structure of the raw data folder inside the processed directory, with the only difference that the text files are replaced by
    # the processed version of the data

    def process(directory):
        # Check if it is a folder and if it contains text files
        if directory.is_dir() and any(file.suffix in [".txt", ".mat"] for file in directory.iterdir()):
            # Make the new folder with the same structure if it does not exist
            processed_folder = interim_data_path.joinpath(name_of_interim_operation).joinpath(
                directory.relative_to(raw_data_path))
            processed_folder.mkdir(parents=True, exist_ok=True)

            # Get the data from the folder
            labeled = True if "training_data" in str(directory) else False
            list_of_data_dicts = load_all_samples_from_folder(directory,
                                                              labelled=labeled,
                                                                file_type=file_type
                                                              )  # A dictionary with the data and metadata
            # Apply the function to each sample in the folder
            for data_dict in list_of_data_dicts:
                processed_data = function_to_apply(data_dict)
                # Write the processed data to a new .txt file in the processed folder
                if file_type == ".txt":
                    np.savetxt(processed_folder.joinpath(data_dict["file_name"]), processed_data)
                elif file_type == ".mat":
                    # Need to change the file_name extension from .mat to .txt
                    data_dict["file_name"] = data_dict["file_name"].split(".mat")[0] + ".txt"
                    np.savetxt(processed_folder.joinpath(data_dict["file_name"]), processed_data)

    directories = list(raw_data_path.glob("**/*"))

    if parallel:
        joblib.Parallel(n_jobs=-1)(joblib.delayed(process)(directory) for directory in directories)
    else:
        for directory in directories:
            process(directory)


def load_all_data_from_interim(directory_with_data):
    path_with_data = interim_data_path.joinpath(directory_with_data)
    directories = [
                   directory for directory in path_with_data.glob("**/*") if
                   # directory.is_dir() and any(file.suffix == ".txt" for file in directory.iterdir())
                   directory.is_dir() and any(file.suffix == ".txt" for file in directory.iterdir())
                   ]

    all_data_dicts = []
    for directory in directories:
        labeled = True if "training_data" in str(directory) else False
        list_of_data_dicts = load_all_samples_from_folder(directory, labelled=labeled)
        all_data_dicts.extend(list_of_data_dicts)
    return all_data_dicts


memory = Memory(location=processed_data_path)
@memory.cache
def get_data(directory_with_data, include_speed_and_load_in_covariates=False):

    # filetype = ".mat" if "csc" in str(directory_with_data) else ".txt"
    list_of_dicts = load_all_data_from_interim(directory_with_data)

    if len(list_of_dicts) == 0:
        raise ValueError("No data found in directory {}".format(directory_with_data))

    df = pd.DataFrame.from_records(list_of_dicts)
    # Let all of the data from the "data" key be its own column named x0, x1, x2, ...
    data_dimension = len(df["data"][0])
    df = df.join(pd.DataFrame(df.pop('data').tolist(), columns=[f"x{i}" for i in range(data_dimension)]))

    # Get only the columns that contains x
    x_cols = [col for col in df.columns if col.startswith("x")]
    # Get all other columns that does not contain x
    meta_data = [col for col in df.columns if not col.startswith("x")]

    if include_speed_and_load_in_covariates:
        x_cols += ["speed", "load"]

    X = df[x_cols]
    meta_data = df[meta_data]
    y = df["health_level"]

    return X, y, meta_data


def sum_everything_together(data_dict):
    return np.sum(data_dict["data"]) * np.ones(3)

def normalized_kurtosis(data_dict):
    acc_data_3_channels = data_dict["data"][:, 0:3].transpose()  #
    return scipy.stats.kurtosis(acc_data_3_channels,axis=1)/np.var(acc_data_3_channels,axis=1)**2

def flattend_everything(data_dict):
    return data_dict["data"].flatten()


