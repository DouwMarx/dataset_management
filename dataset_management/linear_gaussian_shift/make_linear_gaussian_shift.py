import json
import pathlib

import numpy as np
from scipy.optimize import minimize
import pandas as pd

from dataset_management.ultils.write_data_in_standard_format import export_data_to_file_structure


class Process(object):
    def __init__(self, n_features):
        # Set up process attributes
        self.n_features = n_features
        self.healthy_mean = np.zeros(n_features)
        self.healthy_cov = np.eye(
            n_features)  # Can later be extended to have non-identity covariance, Notice that this implies independence between features and PCA should not have an effect.
        self.faulty_cov = np.eye(
            n_features)  # Can later be extended to have non-identity covariance, Notice that this implies independence between features and PCA should not have an effect.

        # Random generation of data
        self.fault_direction = np.random.uniform(-1, 1,
                                                 n_features)  # Random "true" direction in which the fault progresses
        self.fault_direction = self.fault_direction / np.linalg.norm(
            self.fault_direction)  # normalize the fault direction
        self.faulty_test = None  # No data initially assigned to object, but measurements can be taken using take_measurements()
        self.healthy_train = None
        self.healthy_test = None

    def get_healthy_data(self, n_samples):
        return pd.DataFrame(np.random.multivariate_normal(self.healthy_mean, self.healthy_cov, size=n_samples))

    def get_faulty_data_at_severity(self, severity, n_samples):
        return pd.DataFrame(np.random.multivariate_normal(self.healthy_mean + severity * self.fault_direction, self.faulty_cov,
                                             size=n_samples))

    def take_measurements(self, n_samples, severities=None):
        if severities is None:
            severities = [0.5, 1]
        self.healthy_train = self.get_healthy_data(n_samples)
        self.faulty_test = {severity: self.get_faulty_data_at_severity(severity, n_samples) for severity in severities}
        return self.healthy_train, self.faulty_test

    def get_imprecise_fault_directions(self, correlation, n_directions_to_generate=10):
        """
        Used for tests where the true fault direction is not accurately specified.
        To have measurable tests, the relationship between the true fault direction and the imprecise fault directions should be specified.
        We therefore solve an inverse problem where we find an imprecise fault direction with a certain correlation to the true fault direction.
        """

        def objective(x):
            """Objective function to minimize to find random vector with given correlation to fault direction"""
            return (np.dot(x / np.linalg.norm(x), self.fault_direction) - correlation) ** 2

        directions = []
        for i in range(n_directions_to_generate):
            optimisation_startpoint = self.fault_direction + np.random.uniform(-1, 1, self.n_features)
            sol = minimize(objective, optimisation_startpoint)
            # Make sure it is a unit vector
            norm_sol = sol.x / np.linalg.norm(sol.x)
            print("Correlation with true fault direction: ", np.dot(norm_sol, self.fault_direction))
            directions.append(norm_sol)
        return directions


def generate_random_expected_fault_directions(n_fault_directions, n_features, expected_fault_direction_for_true_mode,
                                              max_cor_with_true_fault_direction=None,
                                              include_expected_direction_for_true_mode=True):
    """
    Generate random fault directions but put some constraints on not being perfectly correlated with the true fault direction.
    """
    # Define the expected fault directions
    n_directions_generated = 0
    fault_directions = {}
    while n_directions_generated < n_fault_directions - 1:
        # new_direction = np.random.uniform(-1, 1, n_features)
        new_direction = np.random.normal(size =  n_features) # Use normally distributed data to avoid the corner effect.
        new_direction /= np.linalg.norm(new_direction)
        if max_cor_with_true_fault_direction is not None:
            if np.dot(expected_fault_direction_for_true_mode, new_direction) < max_cor_with_true_fault_direction:
                fault_directions[n_directions_generated] = new_direction
                n_directions_generated += 1
        else:
            fault_directions[n_directions_generated] = new_direction
            n_directions_generated += 1

        # Show percentage complete
        if n_directions_generated % 10 == 0:
            print(f'Data generation {n_directions_generated / n_fault_directions * 100:.0f}% complete')

    if include_expected_direction_for_true_mode: # Only append the true direction when creating expected modes (not unlikely modes)
        fault_directions['true'] = expected_fault_direction_for_true_mode
    return fault_directions


p = Process(n_features=12)
healthy, faulty_dict = p.take_measurements(n_samples=100, severities=[0.5, 1])

export_data_to_file_structure(dataset_name='linear_gaussian_shift',
                              healthy_data=healthy,
                              faulty_data_dict=faulty_dict,
                              export_path=pathlib.Path("/home/douwm/projects/PhD/code/biased_anomaly_detection/data"),
                              metadata={'ground_truth_fault_direction': p.fault_direction.tolist(),
                                        'dataset_name':'linear_gaussian_shift'},)
