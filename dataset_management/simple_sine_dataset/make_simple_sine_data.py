import torch
import numpy as np
from informed_anomaly_detection.utils.pickle_expensive_results import save_dataset
import torch


# TODO: integrate this script into dataset management project

n_features = 10 ** 2 - 1
n_train = 4
t_range = np.linspace(0, 1, n_features).astype("float32")


class DynamicalSystem():

    def __init__(self, sys_properties):
        # The inputs are the deviation from some normal condition
        self.phase = 0 + sys_properties["phase"]
        self.freq = 5 + sys_properties["freq"]
        self.amp= 1 + sys_properties["amp"]
        self.offset= 0 + sys_properties["offset"]
        self.drift= 0 + sys_properties["drift"]
        self.noise= 0.1 + sys_properties["noise"]

    def make_sine(self):
        # Generate the signal corresponding to the given properties
        return np.sin((t_range + self.phase) * 2 * np.pi * self.freq) * self.amp + \
               self.offset + \
               self.drift * t_range + \
               np.random.normal(size=n_features,
                                scale=self.noise)

    def make_many_samples(self,n_samples):
        # Create a dataframe of all samples for a given condition
        d = [self.make_sine() for i in range(n_samples)]
        # df = pd.DataFrame(np.vstack(d).transpose(), index=t_range)
        d = torch.from_numpy(np.vstack(d)).float()
        return d

class DynamicalSystemDataset():
    def __init__(self, n_samples_test, n_samples_train, n_severities):
        self.n_severities = n_severities
        self.n_samples_test = n_samples_test
        self.n_samples_train = n_samples_train

        self.default_sys_properties = {"amp": 0,
                                  "freq": 0,
                                  "offset": 0,
                                  "drift": 0,
                                  "phase": 0,
                                  "noise": 0}

    def make_data_for_different_severities(self,max_severity,mode):
        """ For different severities of the same failure mode, compute many samples for each severity"""
        severities = np.linspace(0,max_severity,self.n_severities)

        severity_dict = {}
        for degree, severity in enumerate(severities):  # Note that the first severity is technically healthy
                sys_properties = self.default_sys_properties
                sys_properties[mode] = severity
                data_for_mode_at_severity = DynamicalSystem(sys_properties).make_many_samples(self.n_samples_test)
                severity_dict[str(degree)] = data_for_mode_at_severity

        return severity_dict

    def make_healthy_data(self):
        """ For different severities of the same failure mode, compute many samples for each severity"""
        healthy_data_dict = {}
        for train_test, sample_size in zip(["train","test"],[self.n_samples_train,self.n_samples_test]):
            sys_properties = self.default_sys_properties
            healthy_data = DynamicalSystem(sys_properties).make_many_samples(sample_size)
            healthy_data_dict[train_test] = healthy_data
        return healthy_data_dict

    def make_data_for_different_failure_modes(self):
        # Deviation from healthy condition when healthy (==0)
        failure_mode_max_severity = {
            "amplitude":1.8,
            "offset":0.4,
            "drift":1.5,
            "phase":0.05*np.pi/4,
            "frequency":0.5}

        failure_modes = list(failure_mode_max_severity.keys())

        data_dict = {}

        # Healthy data
        data_dict["healthy"] = self.make_healthy_data()

        for failure_mode in failure_modes:
            data_for_mode = self.make_data_for_different_severities(failure_mode_max_severity[failure_mode],failure_mode)
            data_dict[failure_mode] = data_for_mode

        return data_dict


def main():
    d = DynamicalSystemDataset(n_samples_test=1000,n_samples_train=2000,n_severities=10).make_data_for_different_failure_modes()
    save_dataset(d,"sine_wave_data")
    return d


if __name__ == "__main__":
    main()


