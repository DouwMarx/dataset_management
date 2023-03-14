import sklearn.datasets as datasets
import pandas as pd
import numpy as np
import pathlib
import json

from dataset_management.ultils.write_data_in_standard_format import export_data_to_file_structure

X,Y=datasets.make_moons(n_samples=1000, noise=0.1, random_state=42)


healthy, faulty = X[Y==0], X[Y==1]

healthy = pd.DataFrame(healthy)
faulty = pd.DataFrame(faulty)

healthy_data = healthy
faulty_data_dict = {1: faulty}

ground_truth_fault_direction = np.array(faulty.mean() - healthy.mean())
ground_truth_fault_direction = ground_truth_fault_direction / np.linalg.norm(ground_truth_fault_direction) # Fault direction is not normalized

name = 'moons'

export_data_to_file_structure(dataset_name=name,
                                healthy_data=healthy,
                                faulty_data_dict=faulty_data_dict,
                                export_path=pathlib.Path("/home/douwm/projects/PhD/code/biased_anomaly_detection/data"),
                                metadata={'ground_truth_fault_direction': list(ground_truth_fault_direction),
                                            'dataset_name': name}
                                )
