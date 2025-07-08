import os
import pathlib
import warnings

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from dataset_management.ultils.write_data_in_standard_format import export_data_to_file_structure
from file_definitions import biased_anomaly_detection_path
import plotly.graph_objects as go

"""
This script organizes the MFPT bearing fault dataset into the standard structure.
The MFPT dataset contains baseline (healthy) conditions and various fault conditions with different loads.
"""

class MFPT(object):
    """
    Used to read and write the MFPT dataset to a standard file structure
    """

    def __init__(self, name="MFPT", overlap=0.0, fault_events_per_segment=8):
        """
        Initialize the MFPT dataset handler
        
        :param name: The name of the dataset
        :param overlap: Overlap between segments (0.0 means no overlap)
        :param fault_events_per_segment: Number of fault events to include in each segment
        """
        self.name = name
        self.overlap = overlap
        self.fault_events_per_segment = fault_events_per_segment
        
        # Path to the raw data
        self.data_path = pathlib.Path(__file__).parent.joinpath("raw_data", "MFPT Fault Data Sets")
        
        if overlap > 0:
            # Raise a warning that splitting should be done responsibly when overlap is used
            warnings.warn("Do not use randomized splitting when overlap is used. This will cause data leakage. Instead, use a fixed split and discard the overlap.")
        
        # Define bearing parameters (from NiceBearing.m)
        self.bearing_params = {
            "roller_diameter": 0.235,
            "pitch_diameter": 1.245,
            "num_elements": 8,
            "contact_angle": 0
        }
        
        # Define dataset metadata
        self.dataset_meta_data = {
            "bearing_params": self.bearing_params,
            "long_name": "MFPT Bearing Fault Dataset",
            "shaft_rate": 25,  # Hz (25 Hz = 1500 RPM)
            "_id": "meta_data"
        }
        
        # Calculate the desired segment length that would ensure a certain number of fault events per segment
        sampling_frequency = 97656 # Hz (Value for first set of outer race fault data)
        outer_fault_freq = self.get_expected_outer_race_fault_frequency()  # Hz
        time_duration_per_event = 1 / outer_fault_freq  # seconds per event
        number_of_samples_per_event =  int(sampling_frequency * time_duration_per_event)  # samples per event
        self.desired_segment_length = int(self.fault_events_per_segment * number_of_samples_per_event)  # samples for all fault events in a segment

        
    def get_expected_outer_race_fault_frequency(self):
        """
        Calculate the expected outer race fault frequency using the bearing fault expressions
        
        :return: Expected outer race fault frequency in Hz
        """
        # Bearing parameters
        rd = self.bearing_params["roller_diameter"]  # roller diameter
        pd = self.bearing_params["pitch_diameter"]   # pitch diameter
        ca = self.bearing_params["contact_angle"]    # contact angle in degrees
        ne = self.bearing_params["num_elements"]     # number of elements
        
        # Shaft rate in Hz
        fr = self.dataset_meta_data["shaft_rate"]
        
        # Calculate outer race fault frequency ratio
        # Formula from GetBearFreqRatio.m: d = 0.5*(1-rdpd*cs)*ne
        rdpd = rd / pd  # roller diameter / pitch diameter
        cs = np.cos(ca * np.pi / 180)  # cosine of contact angle
        outer_race_ratio = 0.5 * (1 - rdpd * cs) * ne
        
        # Calculate outer race fault frequency
        outer_race_freq = outer_race_ratio * fr
        
        return outer_race_freq
    
    def segment_signal(self, signal, segment_length):
        """
        Segment a signal into fixed-length segments with optional overlap
        
        :param signal: The signal to segment
        :param sample_rate: The sample rate of the signal in Hz
        :return: Segmented signal as a numpy array
        """
        
        # Calculate step size based on overlap
        step_size = int(segment_length * (1 - self.overlap))
        
        # Calculate number of segments
        n_segments = (len(signal) - segment_length) // step_size + 1
        
        # Initialize array to store segments
        segments = np.zeros((n_segments, 1, segment_length))
        
        # Extract segments
        for i in range(n_segments):
            start = i * step_size
            end = start + segment_length
            segments[i, 0, :] = signal[start:end]

        print(f"Segmented signal into {n_segments} segments of length {segment_length} samples each.")
        print(f"Each segment should contain approximately {self.fault_events_per_segment} fault events.")
            
        return segments
    
    def load_mat_file(self, file_path):
        """
        Load a .mat file and extract the bearing data
        
        :param file_path: Path to the .mat file
        :return: Tuple of (signal, metadata)
        """
        # Load the .mat file
        mat_data = loadmat(str(file_path))
        
        # Get the bearing struct
        bearing = mat_data['bearing'][0, 0]


        # Check that the signal is does not have more than 1 channel
        if bearing['gs'].ndim > 2:
            raise ValueError(f"Signal in {file_path} has more than 1 channel. Expected a single channel signal.")

        
        # Extract data from the bearing struct
        signal = bearing['gs'].flatten()
        
        print(f"Loaded signal from {file_path} with shape {signal.shape}")
        # Extract metadata
        sample_rate = int(bearing['sr'][0, 0])
        print("Sample rate:", sample_rate)
        load = bearing['load'][0, 0]
        # print("Bearing load:", load)
        shaft_rate = int(bearing['rate'][0, 0])
        # print("Shaft rate:", shaft_rate)
        
        # Convert load to int if possible
        if isinstance(load, (np.ndarray)) and load.dtype.kind in 'SU':
            load = load[0]
        else:
            load = int(load)
        
        # Create metadata dictionary
        metadata = {
            "file_name": os.path.basename(file_path),
            "sample_rate": sample_rate,
            "load": load,
            "shaft_rate": shaft_rate
        }
        
        return signal, metadata
    
    def load_baseline_data(self):
        """
        Load the baseline (healthy) data
        
        :return: List of tuples containing (data, metadata)
        """
        baseline_path = self.data_path.joinpath("1 - Three Baseline Conditions")
        baseline_files = [f for f in os.listdir(baseline_path) if f.endswith('.mat')]
        
        baseline_data = []
        
        for file in baseline_files:
            file_path = baseline_path.joinpath(file)
            signal, metadata = self.load_mat_file(file_path)
            
            # Add condition to metadata
            metadata["condition"] = "baseline"
            
            # Segment the signal
            segments = self.segment_signal(signal, self.desired_segment_length)
            
            baseline_data.append((segments, metadata))
            
        return baseline_data
    
    def load_outer_race_fault_data(self):
        """
        Load only the first set of outer race fault data (270 lbs load)
        
        :return: List of tuples containing (data, metadata)
        """
        # First set of outer race fault data (270 lbs load)
        outer_race_path = self.data_path.joinpath("2 - Three Outer Race Fault Conditions")
        outer_race_files = [f for f in os.listdir(outer_race_path) if f.endswith('.mat')]
        
        outer_race_data = []
        
        # Process first set (270 lbs load, 97656 Hz sample rate)
        for file in outer_race_files:
            file_path = outer_race_path.joinpath(file)
            signal, metadata = self.load_mat_file(file_path)
            
            # Add condition to metadata
            metadata["condition"] = "outer_race_fault"
            # Extract fault number from filename (e.g., OuterRaceFault_1.mat -> 1)
            fault_num = os.path.splitext(file)[0].split('_')[-1]
            metadata["fault_num"] = fault_num
            
            # Segment the signal
            segments = self.segment_signal(signal, self.desired_segment_length)
            
            outer_race_data.append((segments, metadata))
            
        return outer_race_data
    
    def load_data(self):
        """
        Load baseline and outer race fault data from the MFPT dataset
        
        :return: Tuple of (baseline_data, outer_race_fault_data)
        """
        baseline_data = self.load_baseline_data()
        outer_race_fault_data = self.load_outer_race_fault_data()
        
        return baseline_data, outer_race_fault_data


def write_mfpt_to_standard_structure():
    """
    Write the MFPT dataset to the standard file structure.
    For each baseline dataset, include each of the outer race fault datasets as separate fault modes.
    """
    # Initialize the MFPT dataset handler
    mfpt_data = MFPT(name="MFPT", overlap=0.0)
    
    # Load baseline and outer race fault data
    baseline_data, outer_race_fault_data = mfpt_data.load_data()
    
    # Process each baseline dataset
    for i, (healthy_data, healthy_metadata) in enumerate(baseline_data):
        baseline_num = os.path.splitext(healthy_metadata['file_name'])[0].split('_')[-1]
        
        # Create dataset name
        dataset_name = f"MFPT_baseline_{baseline_num}_load_{healthy_metadata['load']}"
        
        # Create faulty data dictionary for this baseline
        faulty_data_dict = {}
        
        # Add each outer race fault dataset as a separate fault mode
        for j, (fault_segments, fault_metadata) in enumerate(outer_race_fault_data):
            fault_num = fault_metadata['fault_num']
            fault_name = f"outer_race_fault_{fault_num}"
            faulty_data_dict[fault_name] = fault_segments
        
        # Update metadata with sample rate
        metadata = mfpt_data.dataset_meta_data.copy()
        metadata.update({"sampling_frequency": healthy_metadata['sample_rate']})
        
        # Add baseline and fault metadata
        metadata["baseline_info"] = {
            "file_name": healthy_metadata['file_name'],
            "load": healthy_metadata['load'],
            "shaft_rate": healthy_metadata['shaft_rate']
        }
        
        metadata["fault_info"] = {
            f"outer_race_fault_{fault_metadata['fault_num']}": {
                "file_name": fault_metadata['file_name'],
                "load": fault_metadata['load'],
                "shaft_rate": fault_metadata['shaft_rate']
            } for _, fault_metadata in outer_race_fault_data
        }
        
        # Calculate outer race fault frequency
        # metadata["expected_fault_frequency"] = {
        #     "outer_race": mfpt_data.get_expected_outer_race_fault_frequency()
        # }
        outer_race_ff = mfpt_data.get_expected_outer_race_fault_frequency()
        metadata["expected_fault_frequencies"] = {mode: outer_race_ff for mode in faulty_data_dict.keys()}

        # Export data to standard structure
        export_data_to_file_structure(
            dataset_name=dataset_name,
            healthy_data=healthy_data,
            faulty_data_dict=faulty_data_dict,
            export_path=biased_anomaly_detection_path,
            metadata=metadata
        )
        
        print(f"Exported dataset: {dataset_name}")


if __name__ == "__main__":
    write_mfpt_to_standard_structure() # Comment out the original function call

    # mfpt_data = MFPT(name="MFPT", overlap=0.0, fault_events_per_segment=100)
    # baseline_data, outer_race_fault_data = mfpt_data.load_data()

    # # Extract 3 healthy signals
    # signals_to_plot = []
    # for i in range(min(3, len(baseline_data))):
    #     # Access the first segment of the healthy data
    #     signal = baseline_data[i][0][0, 0, :]
    #     sample_rate = baseline_data[i][1]['sample_rate']
    #     signals_to_plot.append((signal, sample_rate, f"Healthy Signal {i+1} (Load: {baseline_data[i][1]['load']} lbs)"))

    # # Extract 3 faulty signals
    # for i in range(min(3, len(outer_race_fault_data))):
    #     # Access the first segment of the faulty data
    #     signal = outer_race_fault_data[i][0][0, 0, :]
    #     sample_rate = outer_race_fault_data[i][1]['sample_rate']
    #     signals_to_plot.append((signal, sample_rate, f"Faulty Signal {i+1} (Load: {outer_race_fault_data[i][1]['load']} lbs, Fault: {outer_race_fault_data[i][1]['fault_num']})"))

    # # Create a directory for plots if it doesn't exist
    # plot_dir = pathlib.Path(__file__).parent.joinpath("plots", "psd_plots")
    # plot_dir.mkdir(parents=True, exist_ok=True)

    # fig = go.Figure()

    # for signal, sample_rate, label in signals_to_plot:
    #     frequencies, psd = welch(signal, fs=sample_rate, nperseg=10000)
    #     fig.add_trace(go.Scatter(x=frequencies, y=psd, mode='lines', name=label))
    
    # fig.update_layout(
    #     title="Power Spectral Density of Healthy and Faulty Signals",
    #     xaxis_title="Frequency (Hz)",
    #     yaxis_title="Power/Frequency (dB/Hz)",
    #     hovermode="x unified"
    # )
    # # fig.write_html(str(plot_dir.joinpath("combined_psd_plot.html")))
    # # print(f"Generated combined PSD plot: {str(plot_dir.joinpath('combined_psd_plot.html'))}")
    # fig.show()