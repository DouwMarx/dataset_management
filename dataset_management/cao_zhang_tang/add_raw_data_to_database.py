import pathlib
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.io import loadmat

from dataset_management.ultils.write_data_in_standard_format import export_data_to_file_structure


class CZT(object):
    """
    Used to add the Cao Zhang Tang data
    https://figshare.com/articles/dataset/Gear_Fault_Data/6127874/1?file=11053466
   """

    def __init__(self, percentage_overlap=0.5, required_average_number_of_events_per_rev=30):

        # self.meta_data = {"_id": "meta_data",
        #                                  "signal_length": self.cut_signal_length,
        #                                  "sampling_frequency": self.sampling_frequency,
        #                                  "n_faults_per_revolution": self.n_faults_per_revolution,
        #                                  "dataset_name": "CWR",
        #                                  "cut_signal_length": self.cut_signal_length,
        #                                  })
        self.czt_path = Path("/home/douwm/data/cao_zhang_tang")
        self.health_states = ['healthy', 'missing', 'crack', 'spall', 'chip5a', 'chip4a', 'chip3a', 'chip2a', 'chip1a']
        self.meta_data = {"sampling_frequency": 20480,
                          "dataset_name": "czt",
                          "cut_signal_length": 3600,
                          }

    def load_time_domain_data(self):
        # loop through all .mat files in the  czt path
        file = self.czt_path.joinpath("time_domain.mat")
        data = loadmat(str(file))
        data = data["AccTimeDomain"]  # Data in format (time_samples, n_measurements)
        data = data.transpose()  # Data in format (n_measurements, time_samples)

        data_for_each_class = self.split_data_in_classes(data)

        export_data_to_file_structure(dataset_name="czt",
                                      healthy_data=data_for_each_class["healthy"],
                                      faulty_data_dict= {key: value for key, value in data_for_each_class.items() if key != "healthy"},
                                      export_path= pathlib.Path(
                                          "/home/douwm/projects/PhD/code/biased_anomaly_detection/data"),
                                      metadata= self.meta_data
                                      )

    def split_data_in_classes(self, data):
        """

        From the website:

        types = {'healthy', 'missing', 'crack', 'spall', 'chip5a', 'chip4a', 'chip3a', 'chip2a', 'chip1a'}
        Number of samples per type = 104
        """
        n_per_class = 104
        data_for_each_class = {}
        for i, health_state in enumerate(self.health_states):
            health_state_data = data[i * n_per_class:(i + 1) * n_per_class, :]
            data_for_each_class.update({health_state: health_state_data})

        # Make plotly plot of an example of each class

        healthy_example = data_for_each_class["healthy"][0,:]
        freqs = np.fft.rfftfreq(len(healthy_example), d=1/20480) # Sampling freq listed as 20kHz but more likely to be 20480 Hz?

        fig_freq = go.Figure()
        for class_name, class_data in data_for_each_class.items():
            fft = np.fft.rfft(class_data[0,:])
            fft = np.abs(fft)
            fig_freq.add_trace(go.Scatter(x=freqs, y=fft, name=class_name))
        fig_freq.update_layout(title="Freq domain examples")
        fig_freq.show()

        fig_time = go.Figure()
        for class_name, class_data in data_for_each_class.items():
            sig = class_data[0,:]
            time = np.arange(len(sig))/20480
            fig_time.add_trace(go.Scatter(x=time, y=sig, name=class_name))
        fig_time.update_layout(title="Time domain examples")
        fig_time.show()

        # Save the figures as html
        for fig, name in zip([fig_freq, fig_time], ["freq_domain_examples", "time_domain_examples"]):
            fig.write_html(str(self.czt_path.joinpath(name + ".html")))


        # # Make every array a pd.DataFrame
        # data_for_each_class = {key: pd.DataFrame(value) for key, value in data_for_each_class.items()}

        # Make every array have a channel dimension (batch, channel, time)
        data_for_each_class = {key: np.expand_dims(value, axis=1) for key, value in data_for_each_class.items()}

        return data_for_each_class



        

if __name__ == "__main__":
    czt = CZT()
    czt.load_time_domain_data()