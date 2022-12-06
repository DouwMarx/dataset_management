import pathlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
import pandas as pd

from dataset_management.ultils.write_data_in_standard_format import export_data_to_file_structure

mnist = load_digits()

data = pd.DataFrame(mnist.data)
data['target'] = mnist.target

# Create dictionary of dataframes for each class
data_dict = {}
for i in data.target.unique():
    data_dict[str(i)] = data[data.target == i][data.columns[:-1]]


for health_key, health_value in data_dict.copy().items():
    for fault_key, fault_value in data_dict.copy().items():
        if fault_key != health_key:

            healthy = health_value
            faulty_data_dict = {fault_key: fault_value}

            ground_truth_fault_direction = np.array(fault_value.mean() - health_value.mean())
            ground_truth_fault_direction = ground_truth_fault_direction / np.linalg.norm(ground_truth_fault_direction)


            name = 'digits_health' + health_key + '_fault' + fault_key
            export_data_to_file_structure(dataset_name=name,
                                          healthy_data=healthy,
                                          faulty_data_dict=faulty_data_dict,
                                          export_path=pathlib.Path("/home/douwm/projects/PhD/code/biased_anomaly_detection/data"),
                                          metadata={'ground_truth_fault_direction': list(ground_truth_fault_direction),
                                                    'dataset_name': name}
                                          )

        # plt.figure(health_key + "->" + fault_key)
        # example = ground_truth_fault_direction.reshape(8, 8)
        # plt.imshow(example)