import os
import pathlib
import numpy as np
import scipy.io
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from dataset_management.ultils.write_data_in_standard_format import export_data_to_file_structure
from file_definitions import biased_anomaly_detection_path

class PaderbornBearing:
    def __init__(self, name="PU", overlap=0.0, fault_events_per_segment=20):
        self.name = name
        self.overlap = overlap
        self.fault_events_per_segment = fault_events_per_segment
        self.data_path = pathlib.Path(__file__).parent.joinpath("raw_data")
        
        # Define bearing parameters from the PDF
        self.bearing_params = {
            "inner_diameter": 24.0,     # mm
            "outer_diameter": 33.1,     # mm  
            "pitch_diameter": 28.55,    # mm
            "num_elements": 8,          # Number of rolling elements
            "rolling_element_diameter": 6.75,  # mm
            "contact_angle": 0          # degrees
        }
        
        # Define dataset-specific parameters
        self.dataset_meta_data = {
            "name": self.name,
            "bearing_params": self.bearing_params,
            "long_name": "University of Paderborn Bearing Dataset",
            "_id": "meta_data", 
            "description": "Artificial bearing damage dataset from University of Paderborn - accelerometer signals only",
            "sampling_rate": 64000,  # 64 kHz for vibration signals
            "sampling_frequency": 64000,  # 64 kHz for vibration signals
            "channels": ["accelerometer"],
            "shaft_rate_min": 15,  # Hz, minimum shaft rate (900 RPM / 60 = 15 Hz)
        }
        
        # Operating condition mapping from filename codes
        self.operating_conditions = {
            "N15_M07_F10": {"rpm": 1500, "torque": 0.7, "force": 1000},
            "N09_M07_F10": {"rpm": 900, "torque": 0.7, "force": 1000}, 
            "N15_M01_F10": {"rpm": 1500, "torque": 0.1, "force": 1000},
            "N15_M07_F04": {"rpm": 1500, "torque": 0.7, "force": 400}
        }
        
        # Calculate segment length based on lowest RPM (900) for consistency across all conditions
        # Slowest RPM is used to ensure at least a certain number of fault events per segment
        self.sampling_frequency = 64000  # Hz
        self.calculate_segment_length()
        
        # Bearing condition mapping
        self.condition_mapping = {
            "K001": "healthy",
            "K002": "healthy", 
            "K003": "healthy",
            "K004": "healthy",
            "K005": "healthy",
            "K006": "healthy",
            "KA01": "artificial_damage_A01",
            "KA03": "artificial_damage_A03",
            "KA04": "artificial_damage_A04",
            "KA05": "artificial_damage_A05",
            "KA06": "artificial_damage_A06",
            "KA07": "artificial_damage_A07",
            "KA08": "artificial_damage_A08",
            "KA09": "artificial_damage_A09",
            "KA15": "artificial_damage_A15",
            "KA16": "artificial_damage_A16",
            "KA22": "artificial_damage_A22",
            "KA30": "artificial_damage_A30",
            "KB23": "artificial_damage_B23",
            "KB24": "artificial_damage_B24",
            "KB25": "artificial_damage_B25",
            "KB26": "artificial_damage_B26",
            "KB27": "artificial_damage_B27",
            "KI01": "artificial_damage_I01",
            "KI03": "artificial_damage_I03",
            "KI04": "artificial_damage_I04",
            "KI05": "artificial_damage_I05",
            "KI07": "artificial_damage_I07",
            "KI08": "artificial_damage_I08",
            "KI14": "artificial_damage_I14",
            "KI16": "artificial_damage_I16",
            "KI17": "artificial_damage_I17",
            "KI18": "artificial_damage_I18",
            "KI21": "artificial_damage_I21"
        }
    
    def get_expected_outer_race_fault_frequency(self):
        """
        Calculate the expected outer race fault frequency using bearing fault expressions.
        Outer race fault is typically the slowest frequency.
        
        :return: Expected outer race fault frequency in Hz
        """
        # Convert bearing parameters to consistent units
        rd = self.bearing_params["rolling_element_diameter"]  # mm
        pd = self.bearing_params["pitch_diameter"]            # mm
        ca = self.bearing_params["contact_angle"]             # degrees
        ne = self.bearing_params["num_elements"]              # number of elements
        
        # Shaft rate in Hz (use minimum RPM for consistent segmentation)
        fr = self.dataset_meta_data["shaft_rate_min"]
        
        # Calculate outer race fault frequency ratio
        # Formula: BPFO = (N/2) * (1 - (d/D) * cos(α)) * fr
        # where N = number of balls, d = ball diameter, D = pitch diameter, α = contact angle
        rdpd = rd / pd  # roller diameter / pitch diameter ratio
        cs = np.cos(ca * np.pi / 180)  # cosine of contact angle
        outer_race_ratio = 0.5 * ne * (1 - rdpd * cs)
        
        # Calculate outer race fault frequency
        outer_race_freq = outer_race_ratio * fr
        
        print(f"Calculated outer race fault frequency: {outer_race_freq:.2f} Hz")
        return outer_race_freq
    
    def get_expected_inner_race_fault_frequency(self):
        """
        Calculate the expected inner race fault frequency.
        
        :return: Expected inner race fault frequency in Hz
        """
        # Convert bearing parameters
        rd = self.bearing_params["rolling_element_diameter"]  # mm
        pd = self.bearing_params["pitch_diameter"]            # mm  
        ca = self.bearing_params["contact_angle"]             # degrees
        ne = self.bearing_params["num_elements"]              # number of elements
        
        # Shaft rate in Hz (use minimum RPM for consistent segmentation)
        fr = self.dataset_meta_data["shaft_rate_min"]
        
        # Calculate inner race fault frequency ratio
        # Formula: BPFI = (N/2) * (1 + (d/D) * cos(α)) * fr
        rdpd = rd / pd  # roller diameter / pitch diameter ratio
        cs = np.cos(ca * np.pi / 180)  # cosine of contact angle
        inner_race_ratio = 0.5 * ne * (1 + rdpd * cs)
        
        # Calculate inner race fault frequency
        inner_race_freq = inner_race_ratio * fr
        
        return inner_race_freq
    
    def calculate_segment_length(self):
        """
        Calculate the desired segment length based on outer race fault frequency
        to ensure each segment contains a specified number of fault events.
        """
        # Get outer race fault frequency (slowest, so good for capturing fault events)
        outer_fault_freq = self.get_expected_outer_race_fault_frequency()
        
        # Calculate time duration per fault event
        time_duration_per_event = 1 / outer_fault_freq  # seconds per event
        
        # Calculate number of samples per fault event
        samples_per_event = int(self.sampling_frequency * time_duration_per_event)
        
        # Calculate desired segment length for specified number of fault events
        self.desired_segment_length = int(self.fault_events_per_segment * samples_per_event)
        
        print(f"Segment configuration:")
        print(f"  - Outer race fault frequency: {outer_fault_freq:.2f} Hz")
        print(f"  - Samples per fault event: {samples_per_event}")
        print(f"  - Desired segment length: {self.desired_segment_length} samples")
        print(f"  - Fault events per segment: {self.fault_events_per_segment}")
        print(f"  - Segment duration: {self.desired_segment_length/self.sampling_frequency:.3f} seconds")
    
    def parse_operating_condition(self, filename):
        """
        Extract operating condition from filename (e.g., N15_M07_F10_K001_13.mat -> N15_M07_F10)
        """
        parts = filename.replace('.mat', '').split('_')
        if len(parts) >= 3:
            return f"{parts[0]}_{parts[1]}_{parts[2]}"
        return None
    
    def load_mat_file(self, file_path):
        """Load a single MAT file and extract vibration signals."""
        try:
            data = scipy.io.loadmat(str(file_path))
            
            # Get the main variable (skip MATLAB metadata)
            main_var = None
            for key, value in data.items():
                if not key.startswith('__'):
                    main_var = key
                    break
            
            if not main_var:
                return None, None
            
            main_data = data[main_var]
            
            # Extract Y (measurement channels) data
            y_data = main_data['Y'][0, 0]
            
            # Get channel names to identify vibration channels
            names = y_data['Name']
            data_field = y_data['Data']
            
            # Find the vibration channel (should be channel 6: 'vibration_1')
            vibration_signal = None
            
            for i in range(data_field.shape[1]):
                channel_name = names[0, i][0] if names[0, i].size > 0 else f"channel_{i}"
                
                if 'vibration' in channel_name.lower():
                    signal_data = data_field[0, i]
                    vibration_signal = signal_data.flatten()
                    # print(f"  Found accelerometer: {channel_name} with {signal_data.size} samples")
                    break
            
            if vibration_signal is not None:
                # Return single channel vibration signal
                combined_signal = vibration_signal.reshape(1, -1)
                # print(f"  Extracted accelerometer signal shape: {combined_signal.shape}")
                return combined_signal, self.dataset_meta_data
            else:
                print(f"  No accelerometer signal found in {file_path.name}")
                return None, None
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            import traceback
            traceback.print_exc()
            
        return None, None
    
    def load_healthy_data(self):
        """Load healthy bearing data grouped by operating condition."""
        healthy_data_by_condition = {}
        
        for condition_dir in self.data_path.iterdir():
            if condition_dir.is_dir() and self.condition_mapping.get(condition_dir.name) == "healthy":
                print(f"Loading healthy data from {condition_dir.name}")
                
                mat_files = list(condition_dir.rglob("*.mat"))
                for mat_file in mat_files:  # Process all files
                    # Parse operating condition from filename
                    operating_condition = self.parse_operating_condition(mat_file.name)
                    if operating_condition and operating_condition in self.operating_conditions:
                        signal, metadata = self.load_mat_file(mat_file)
                        if signal is not None:
                            # Segment the signal
                            segments = self.segment_signal(signal)
                            if segments.shape[0] > 0:
                                if operating_condition not in healthy_data_by_condition:
                                    healthy_data_by_condition[operating_condition] = []
                                healthy_data_by_condition[operating_condition].append((segments, metadata))
                        
        return healthy_data_by_condition
    
    def load_faulty_data(self):
        """Load faulty bearing data organized by fault type and operating condition."""
        faulty_data_by_condition = {}
        
        for condition_dir in self.data_path.iterdir():
            if condition_dir.is_dir() and condition_dir.name in self.condition_mapping:
                condition = self.condition_mapping[condition_dir.name]
                
                if condition != "healthy":
                    print(f"Loading faulty data from {condition_dir.name} -> {condition}")
                    
                    mat_files = list(condition_dir.rglob("*.mat"))
                    
                    for mat_file in mat_files:  # Process all files
                        # Parse operating condition from filename
                        operating_condition = self.parse_operating_condition(mat_file.name)
                        if operating_condition and operating_condition in self.operating_conditions:
                            signal, metadata = self.load_mat_file(mat_file)
                            if signal is not None:
                                # Segment the signal
                                segments = self.segment_signal(signal)
                                if segments.shape[0] > 0:
                                    if operating_condition not in faulty_data_by_condition:
                                        faulty_data_by_condition[operating_condition] = {}
                                    if condition not in faulty_data_by_condition[operating_condition]:
                                        faulty_data_by_condition[operating_condition][condition] = []
                                    faulty_data_by_condition[operating_condition][condition].append(segments)
        
        # Combine segments for each fault type within each operating condition
        for operating_condition in faulty_data_by_condition:
            for condition in faulty_data_by_condition[operating_condition]:
                fault_segments = faulty_data_by_condition[operating_condition][condition]
                if fault_segments:
                    combined_segments = np.concatenate(fault_segments, axis=0)
                    faulty_data_by_condition[operating_condition][condition] = combined_segments
        
        return faulty_data_by_condition
    
    def segment_signal(self, signal):
        """Segment time series data into fixed-length segments based on fault frequencies."""
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        
        n_channels, signal_length = signal.shape
        
        # Use the calculated segment length based on outer race fault frequency
        segment_length = self.desired_segment_length
        step_size = int(segment_length * (1 - self.overlap))
        
        n_segments = max(0, (signal_length - segment_length) // step_size + 1)
        
        if n_segments == 0:
            print(f"  Warning: Signal too short ({signal_length} samples) for desired segment length ({segment_length} samples)")
            # If signal is too short, use the whole signal as one segment
            if signal_length >= 1000:  # Minimum 1000 samples
                padded_signal = np.zeros((n_channels, segment_length))
                padded_signal[:, :signal_length] = signal
                return padded_signal.reshape(1, n_channels, segment_length)
            else:
                return np.array([]).reshape(0, n_channels, segment_length)
        
        segments = np.zeros((n_segments, n_channels, segment_length))
        
        for i in range(n_segments):
            start = i * step_size
            end = start + segment_length
            if end <= signal_length:
                segments[i, :, :] = signal[:, start:end]
        
        print(f"  Segmented signal into {n_segments} segments of {segment_length} samples each")
        return segments

def write_to_standard_structure():
    """Process and export Paderborn bearing data to standard structure, separated by operating condition."""
    print("=== Processing Paderborn Bearing Dataset ===")
    
    # Initialize dataset handler
    dataset = PaderbornBearing()
    
    # Load healthy data grouped by operating condition
    print("\n--- Loading Healthy Data ---")
    healthy_data_by_condition = dataset.load_healthy_data()
    print(f"Loaded healthy data for {len(healthy_data_by_condition)} operating conditions:")
    for condition, data_list in healthy_data_by_condition.items():
        print(f"  {condition}: {len(data_list)} files")
    
    # Load faulty data grouped by operating condition
    print("\n--- Loading Faulty Data ---") 
    faulty_data_by_condition = dataset.load_faulty_data()
    print(f"Loaded faulty data for {len(faulty_data_by_condition)} operating conditions:")
    for condition, fault_dict in faulty_data_by_condition.items():
        print(f"  {condition}: {len(fault_dict)} fault types")
        for fault_type, segments in fault_dict.items():
            print(f"    {fault_type}: {segments.shape[0]} segments")
    
    # Export separate datasets for each operating condition
    print("\n--- Exporting to Standard Structure ---")
    
    # Find all operating conditions that have both healthy and faulty data
    all_conditions = set(healthy_data_by_condition.keys()) | set(faulty_data_by_condition.keys())
    
    for operating_condition in all_conditions:
        print(f"\nProcessing operating condition: {operating_condition}")
        
        # Get operating condition details
        condition_details = dataset.operating_conditions.get(operating_condition, {})
        rpm = condition_details.get('rpm', 'unknown')
        torque = condition_details.get('torque', 'unknown')
        force = condition_details.get('force', 'unknown')
        
        # Create dataset name with operating condition
        dataset_name = f"{dataset.name}_{operating_condition}_RPM{rpm}_T{torque}_F{force}"
        
        # Prepare healthy data for this condition
        combined_healthy = None
        if operating_condition in healthy_data_by_condition:
            healthy_data_list = healthy_data_by_condition[operating_condition]
            if healthy_data_list:
                all_healthy_segments = []
                for segments, _ in healthy_data_list:
                    all_healthy_segments.append(segments)
                combined_healthy = np.concatenate(all_healthy_segments, axis=0)
                print(f"  Healthy segments: {combined_healthy.shape[0]}")
        
        # Prepare faulty data for this condition
        faulty_data_dict = {}
        if operating_condition in faulty_data_by_condition:
            faulty_data_dict = faulty_data_by_condition[operating_condition]
        
        # Only export if we have healthy data
        if combined_healthy is not None:
            # Create metadata for this specific operating condition
            condition_metadata = dataset.dataset_meta_data.copy()
            condition_metadata.update({
                'operating_condition': operating_condition,
                'rpm': rpm,
                'torque_nm': torque,
                'radial_force_n': force,
                'description': f"Paderborn bearing dataset - Operating condition {operating_condition} (RPM: {rpm}, Torque: {torque} Nm, Force: {force} N)"
            })
            
            # Export to standard structure
            export_data_to_file_structure(
                dataset_name=dataset_name,
                healthy_data=combined_healthy,
                faulty_data_dict=faulty_data_dict,
                export_path=biased_anomaly_detection_path,
                metadata=condition_metadata
            )
            
            print(f"  Exported dataset: {dataset_name}")
        else:
            print(f"  Skipping {operating_condition} - no healthy data found")
    
    print("\n✓ All Paderborn bearing datasets exported successfully!")

if __name__ == "__main__":
    write_to_standard_structure()