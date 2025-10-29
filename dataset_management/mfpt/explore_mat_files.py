import os
import pathlib
from scipy.io import loadmat
import numpy as np

"""
This script explores the structure of the .mat files in the MFPT dataset
to understand their content before implementing the full data processing.
"""


def explore_mat_file(file_path):
    """
    Load a .mat file and print its keys and basic information

    :param file_path: Path to the .mat file
    """
    print(f"\nExploring file: {file_path}")

    # Load the .mat file
    mat_data = loadmat(str(file_path))

    # Print the keys (variables) in the .mat file
    print("Keys in the .mat file:")
    for key in mat_data.keys():
        if not key.startswith("__"):  # Skip metadata keys
            print(f"  - {key}")

            # Print information about the variable
            var = mat_data[key]
            print(f"    Type: {type(var)}")
            print(f"    Shape: {var.shape}")

            # If it's a struct, print its fields
            if var.dtype.kind == "V":
                print(f"    Fields: {var.dtype.names}")

                # Print sample of the first struct
                if len(var) > 0:
                    print(f"    First struct content:")
                    for field in var.dtype.names:
                        print(f"      {field}: {var[0][field]}")

            # If it's a numeric array, print some statistics
            elif var.dtype.kind in "iuf":
                print(f"    Min: {np.min(var)}")
                print(f"    Max: {np.max(var)}")
                print(f"    Mean: {np.mean(var)}")
                print(f"    Std: {np.std(var)}")


def explore_directory(directory_path):
    """
    Explore all .mat files in a directory

    :param directory_path: Path to the directory
    """
    print(f"Exploring directory: {directory_path}")

    # Get all .mat files in the directory
    mat_files = [f for f in os.listdir(directory_path) if f.endswith(".mat")]

    if not mat_files:
        print("No .mat files found in the directory.")
        return

    print(f"Found {len(mat_files)} .mat files:")
    for file in mat_files:
        file_path = os.path.join(directory_path, file)
        explore_mat_file(file_path)


if __name__ == "__main__":
    # Path to the MFPT dataset
    base_path = pathlib.Path(__file__).parent.joinpath(
        "raw_data", "MFPT Fault Data Sets"
    )

    # Explore baseline data
    baseline_path = base_path.joinpath("1 - Three Baseline Conditions")
    explore_directory(baseline_path)

    # Explore outer race fault data (first set)
    outer_race_path_1 = base_path.joinpath("2 - Three Outer Race Fault Conditions")
    explore_directory(outer_race_path_1)

    # Explore outer race fault data (second set)
    outer_race_path_2 = base_path.joinpath("3 - Seven More Outer Race Fault Conditions")
    explore_directory(outer_race_path_2)

    # Explore inner race fault data
    inner_race_path = base_path.joinpath("4 - Seven Inner Race Fault Conditions")
    explore_directory(inner_race_path)

    # Explore analyses data
    analyses_path = base_path.joinpath("5 - Analyses")
    explore_directory(analyses_path)
