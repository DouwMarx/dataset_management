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
for label in data.target.unique():
    data_for_label = data[data.target == label]
    data_without_label = data_for_label.drop('target', axis=1)

    print(data_without_label.max().max())
    # Add noise to the data and clip it to the range [0, 255]
    # noisy_data = data_without_label + np.random.normal(0, 0.2*16, data_without_label.shape)
    # noisy_data = np.clip(noisy_data, 0, 16)
    # data_dict[str(label)] = noisy_data

    data_dict[str(label)] = data_without_label


# Currently only creating comparissons between different classes (not including unseen classes)
for health_key, health_value in data_dict.copy().items():
    for fault_key, fault_value in data_dict.copy().items():
        if fault_key != health_key:

            healthy = health_value
            faulty_data_dict = {fault_key: fault_value}

            ground_truth_fault_direction = np.array(fault_value.mean() - health_value.mean())
            # ground_truth_fault_direction = ground_truth_fault_direction / np.linalg.norm(ground_truth_fault_direction) # Fault direction is not normalized


            name = 'digits_health' + health_key + '_fault' + fault_key
            export_data_to_file_structure(dataset_name=name,
                                          healthy_data=healthy,
                                          faulty_data_dict=faulty_data_dict,
                                          export_path=pathlib.Path("/home/douwm/projects/PhD/code/biased_anomaly_detection/data"),
                                          metadata={'ground_truth_fault_direction': list(ground_truth_fault_direction),
                                                    'dataset_name': name}
                                          )

# plt.figure(health_key)
# example = healthy.iloc[0].values.reshape(8, 8)
# plt.imshow(example)
#
# plt.figure(health_key + "->" + fault_key)
# example = ground_truth_fault_direction.reshape(8, 8)
# plt.imshow(example)